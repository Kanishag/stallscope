import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import core

app = FastAPI(title="StallScope API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

db = None
MODEL = None


def connect():
    """Connect to MongoDB Atlas and load the trained model. The app still works without it."""
    global db, MODEL
    uri = os.environ.get("MONGO_URI")
    if not uri:
        return
    try:
        import certifi
        from pymongo import MongoClient
        client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=8000)
        db = client["stallscope"]
        doc = db["model"].find_one({"_id": "forest"})
        MODEL = doc["model"] if doc else None
    except Exception as e:
        print("MongoDB not available:", e)
        db = None


connect()


class Req(BaseModel):
    code: str
    forwarding: bool = True


@app.get("/")
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "database": db is not None, "model_loaded": MODEL is not None}


@app.post("/analyze")
def analyze(req: Req):
    try:
        res = core.analyze(req.code, req.forwarding, MODEL)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if db is not None:
        try:
            db["usage"].insert_one({"t": time.time(), "n": res["instructions"], "forwarding": req.forwarding,
                                    "stalls": res["stalls"], "cycles": res["cycles"]})
        except Exception:
            pass
    return res


@app.get("/stats")
def stats():
    out = {"dataset": None, "model": None, "analyses": 0}
    if db is not None:
        try:
            out["dataset"] = db["summary"].find_one({"_id": "dataset"}, {"_id": 0})
            out["analyses"] = db["usage"].count_documents({})
            if MODEL:
                out["model"] = {k: v for k, v in MODEL.items() if k.endswith("_mae") or k.endswith("_r2")}
        except Exception:
            pass
    return out
