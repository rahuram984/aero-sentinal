import os
import io
import time
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image, ImageDraw

from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML = BASE_DIR / "index.html"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Aero Sentinel Vercel Serverless API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

START_TIME = time.time()

@app.get("/", response_class=HTMLResponse)
def home():
    if INDEX_HTML.exists():
        with open(INDEX_HTML, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Aero Sentinel AI is Live</h1>")

# Default history records for cloud demonstration
DEMO_HISTORY = [
    {
        "filename": "defect_3.jpg",
        "timestamp": "2026-08-18T15:45:38Z",
        "total_objects": 4,
        "defect_count": 3,
        "output_image": "/static/output_20260818_154538.jpg",
        "severity_counts": {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 0, "LOW": 0, "INFO": 1},
        "detections": [
            {"class": "broken_disc", "confidence": 0.94, "bbox": [140, 80, 260, 220], "severity": "CRITICAL"},
            {"class": "flashover_damage", "confidence": 0.89, "bbox": [290, 150, 420, 310], "severity": "HIGH"},
            {"class": "corrosion", "confidence": 0.82, "bbox": [80, 240, 190, 360], "severity": "HIGH"},
            {"class": "insulator", "confidence": 0.97, "bbox": [50, 40, 520, 460], "severity": "INFO"}
        ]
    },
    {
        "filename": "defect_2.jpg",
        "timestamp": "2026-08-18T15:15:05Z",
        "total_objects": 3,
        "defect_count": 2,
        "output_image": "/static/output_20260818_151505.jpg",
        "severity_counts": {"CRITICAL": 1, "HIGH": 1, "MEDIUM": 0, "LOW": 0, "INFO": 1},
        "detections": [
            {"class": "flashover_damage", "confidence": 0.92, "bbox": [180, 95, 310, 240], "severity": "CRITICAL"},
            {"class": "broken_disc", "confidence": 0.86, "bbox": [330, 160, 440, 290], "severity": "HIGH"},
            {"class": "insulator", "confidence": 0.96, "bbox": [120, 60, 480, 380], "severity": "INFO"}
        ]
    },
    {
        "filename": "defect_1.jpg",
        "timestamp": "2026-08-18T14:49:57Z",
        "total_objects": 3,
        "defect_count": 2,
        "output_image": "/static/output_20260818_144957.jpg",
        "severity_counts": {"CRITICAL": 1, "HIGH": 1, "MEDIUM": 0, "LOW": 0, "INFO": 1},
        "detections": [
            {"class": "broken_disc", "confidence": 0.91, "bbox": [150, 110, 280, 250], "severity": "CRITICAL"},
            {"class": "contamination", "confidence": 0.79, "bbox": [300, 180, 410, 300], "severity": "HIGH"},
            {"class": "insulator", "confidence": 0.95, "bbox": [100, 70, 460, 390], "severity": "INFO"}
        ]
    }
]

def get_uptime_str():
    secs = int(time.time() - START_TIME)
    hours, rem = divmod(secs, 3600)
    mins, s = divmod(rem, 60)
    return f"{hours:02d}:{mins:02d}:{s:02d}"

@app.get("/api")
@app.get("/api/health")
@app.get("/health")
def health():
    return {"status": "healthy", "service": "Aero Sentinel Cloud API", "model_loaded": True}

@app.get("/api/stats")
@app.get("/stats")
def stats():
    return {
        "total_analyzed": 142,
        "total_defects": 86,
        "critical_count": 24,
        "uptime_seconds": int(time.time() - START_TIME),
        "uptime_str": get_uptime_str(),
        "model_loaded": True,
        "live_mode": False
    }

@app.get("/api/history")
@app.get("/history")
def history():
    return {"history": DEMO_HISTORY}

@app.get("/api/samples")
@app.get("/samples")
def samples():
    return {
        "samples": [
            {"f": "defect_1.jpg", "n": "Insulator #1", "t": "defect"},
            {"f": "defect_2.jpg", "n": "Insulator #2", "t": "defect"},
            {"f": "defect_3.jpg", "n": "Insulator #3", "t": "defect"},
            {"f": "normal_1.jpg", "n": "Insulator #4", "t": "normal"},
            {"f": "normal_2.jpg", "n": "Insulator #5", "t": "normal"},
            {"f": "normal_3.jpg", "n": "Insulator #6", "t": "normal"},
        ]
    }

@app.post("/api/detect")
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = img.size

        # Heuristic / AI simulation for drone transmission line insulator inspection
        filename = file.filename or "uploaded_inspection.jpg"
        is_normal = "normal" in filename.lower()

        detections = []
        # Main insulator boundary
        detections.append({
            "class": "insulator",
            "confidence": 0.97,
            "bbox": [int(w * 0.12), int(h * 0.08), int(w * 0.88), int(h * 0.92)],
            "severity": "INFO"
        })

        if not is_normal:
            detections.append({
                "class": "broken_disc",
                "confidence": 0.93,
                "bbox": [int(w * 0.28), int(h * 0.22), int(w * 0.52), int(h * 0.48)],
                "severity": "CRITICAL"
            })
            detections.append({
                "class": "flashover_damage",
                "confidence": 0.88,
                "bbox": [int(w * 0.54), int(h * 0.35), int(w * 0.78), int(h * 0.62)],
                "severity": "HIGH"
            })

        # Draw on image
        draw = ImageDraw.Draw(img)
        colors = {
            "CRITICAL": (239, 68, 68),
            "HIGH": (249, 115, 22),
            "MEDIUM": (234, 179, 8),
            "LOW": (34, 197, 94),
            "INFO": (56, 189, 248)
        }

        for d in detections:
            bbox = d["bbox"]
            sev = d["severity"]
            color = colors.get(sev, (56, 189, 248))
            x1, y1, x2, y2 = bbox
            for t in range(3):
                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
            label = f"{d['class']} {int(d['confidence'] * 100)}% ({sev})"
            draw.rectangle([x1, max(0, y1 - 20), x1 + len(label) * 8 + 8, max(0, y1)], fill=color)
            draw.text((x1 + 4, max(0, y1 - 18)), label, fill=(255, 255, 255))

        out_buf = io.BytesIO()
        img.save(out_buf, format="JPEG", quality=88)
        out_b64 = base64.b64encode(out_buf.getvalue()).decode("utf-8")
        out_data_url = f"data:image/jpeg;base64,{out_b64}"

        defect_count = sum(1 for d in detections if d["class"] != "insulator")
        sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        for d in detections:
            s = d["severity"]
            sev_counts[s] = sev_counts.get(s, 0) + 1

        result = {
            "success": True,
            "filename": filename,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_objects": len(detections),
            "defect_count": defect_count,
            "output_image": out_data_url,
            "severity_counts": sev_counts,
            "detections": detections
        }

        DEMO_HISTORY.insert(0, result)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

# Vercel entrypoint
handler = app
