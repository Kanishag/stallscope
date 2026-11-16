"""StallScope core: parse assembly, simulate a 5-stage pipeline, search the best order, predict with the ML forest."""
import re

REG = re.compile(r"^r(\d+)$")
FEATURES = ["n", "n_lw", "n_sw", "raw1", "raw2", "raw3", "loaduse1", "chain", "stalls_fw", "stalls_nofw"]


def _reg(x):
    m = REG.match(x or "")
    return int(m.group(1)) if m else None


def parse_program(code):
    prog = []
    for raw in code.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        t = [x for x in re.split(r"[,\s()]+", line) if x]
        op, r = t[0].lower(), t[1:]
        try:
            if op == "lw":
                rd, srcs = _reg(r[0]), [_reg(r[2])]
            elif op == "sw":
                rd, srcs = None, [_reg(r[0]), _reg(r[2])]
            elif op in ("beq", "bne"):
                rd, srcs = None, [_reg(r[0]), _reg(r[1])]
            else:
                rd, srcs = _reg(r[0]), [_reg(r[1]), _reg(r[2])]
        except IndexError:
            raise ValueError("Cannot read this line: " + line)
        srcs = tuple(s for s in srcs if s)   # drops immediates and r0
        if rd == 0:
            rd = None
        prog.append((op, rd, srcs, line))
    return prog


def simulate(prog, fw):
    """Returns (stall cycles, total cycles, per-instruction info)."""
    ready, info = {}, []
    pf = pd = pE = 0
    stalls = 0
    E = 0
    for i, (op, rd, srcs, text) in enumerate(prog):
        f = 1 if i == 0 else max(pf + 1, pd)
        d = max(f + 1, pE)
        base = d + 1
        need, why = base, None
        for r in srcs:
            if r in ready:
                Ej, ld, j = ready[r]
                av = (Ej + (2 if ld else 1)) if fw else Ej + 3
                if av > need:
                    need, why = av, (r, j)
        stalls += need - base
        E = need
        info.append((need - base, why))
        if rd:
            ready[rd] = (E, op == "lw", i)
        pf, pd, pE = f, d, E
    return stalls, E + 2, info


def chain_len(prog):
    n = len(prog)
    L = [1] * n
    for j in range(n):
        for i in range(j):
            if prog[i][1] and prog[i][1] in prog[j][2]:
                L[j] = max(L[j], L[i] + 1)
    return max(L)


def features(prog):
    n = len(prog)
    n_lw = sum(p[0] == "lw" for p in prog)
    n_sw = sum(p[0] == "sw" for p in prog)
    raw = [0, 0, 0]
    lu1 = 0
    for i, p in enumerate(prog):
        for dist in (1, 2, 3):
            if i - dist >= 0:
                prd = prog[i - dist][1]
                if prd and prd in p[2]:
                    raw[dist - 1] += 1
                    if dist == 1 and prog[i - 1][0] == "lw":
                        lu1 += 1
    return n, n_lw, n_sw, raw[0], raw[1], raw[2], lu1, chain_len(prog)


def _deps(prog):
    mem = lambda x: x[0] in ("lw", "sw")
    preds = []
    for j, b in enumerate(prog):
        p = []
        for i in range(j):
            a = prog[i]
            if ((a[1] and a[1] in b[2]) or (b[1] and b[1] in a[2]) or (a[1] and a[1] == b[1]) or
                    (mem(a) and mem(b) and (a[0] == "sw" or b[0] == "sw")) or
                    a[0] in ("beq", "bne") or b[0] in ("beq", "bne")):
                p.append(i)
        preds.append(p)
    return preds


def best_order(prog, fw):
    """Exact search over every order that keeps the program's meaning. Up to 8 instructions."""
    n = len(prog)
    preds = _deps(prog)
    best = [simulate(prog, fw)[1], list(range(n))]
    used, cur = [False] * n, []

    def go():
        if len(cur) == n:
            c = simulate([prog[i] for i in cur], fw)[1]
            if c < best[0]:
                best[0], best[1] = c, cur[:]
            return
        for i in range(n):
            if used[i] or not all(used[p] for p in preds[i]):
                continue
            used[i] = True
            cur.append(i)
            go()
            cur.pop()
            used[i] = False

    go()
    return best[0], best[1]


def predict_forest(trees, x):
    s = 0.0
    for t in trees:
        i = 0
        while t["l"][i] != -1:
            i = t["l"][i] if x[t["f"][i]] <= t["t"][i] else t["r"][i]
        s += t["v"][i]
    return s / len(trees)


def analyze(code, fw=True, model=None):
    prog = parse_program(code)
    if not prog:
        raise ValueError("Add at least one instruction, for example: add r1, r2, r3")
    if len(prog) > 40:
        raise ValueError("Please keep the program under 40 instructions.")
    n = len(prog)
    s_fw, c_fw, info_fw = simulate(prog, True)
    s_no, c_no, info_no = simulate(prog, False)
    stalls, cycles, info = (s_fw, c_fw, info_fw) if fw else (s_no, c_no, info_no)
    hazards = []
    for i, (st, why) in enumerate(info):
        if st > 0 and why:
            hazards.append({"line": i + 1, "text": prog[i][3], "reg": "r%d" % why[0], "from_line": why[1] + 1, "stall": st})
    out = {
        "instructions": n, "forwarding": fw, "stalls": stalls, "cycles": cycles,
        "cpi": round(cycles / n, 2), "speedup": round(5 * n / cycles, 2),
        "stalls_if_forwarding": s_fw, "stalls_if_no_forwarding": s_no,
        "hazards": hazards, "ml": None, "exact": None,
    }
    if n <= 8:
        b, order = best_order(prog, fw)
        out["exact"] = {"best_cycles": b, "saved": cycles - b, "order": [prog[i][3] for i in order]}
    if model:
        key = "saved_fw" if fw else "saved_nofw"
        if key in model:
            x = list(features(prog)) + [s_fw, s_no]
            pred = max(0.0, predict_forest(model[key], x))
            out["ml"] = {"predicted_saved": round(pred, 2), "typical_error": model.get(key + "_mae"),
                         "in_training_range": 6 <= n <= 8}
    return out
