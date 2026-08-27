# ⚡ Aero Sentinel — AI Insulator Defect Detection
### Smart India Hackathon 2026 • Team ID: `SIH26201` • Theme: Robotics & Drones

> **AI-powered drone defect inspection for power transmission lines.** Built with YOLOv11 deep learning, FastAPI backend, real-time bounding box telemetry, and automated A4 PDF inspection reports.

---

## 🌟 Key Features

- 🛰️ **Real-Time Drone Defect Inspection:** Automated detection of broken insulator discs, missing caps, flashover burns, and corrosion.
- 🎯 **Dynamic Confidence Filtering:** Real-time interactive slider to adjust prediction thresholds on the fly.
- 🚨 **Multi-Tier Severity Grading:** Automated risk classification (**CRITICAL**, **HIGH**, **MEDIUM**, **LOW**, **INFO**).
- 📄 **1-Click A4 PDF Reports:** Generates professional engineering inspection reports with bounding box coordinates, timestamps, and severity summary.
- ⚡ **Live YOLO Inference:** Real-time predictions (~15–20ms per frame) on any uploaded drone footage.
- 🌓 **Modern Glassmorphic UI:** Responsive dark/light theme, live inspection dashboard, and test gallery.

---

## 📁 Repository Structure

```
aero-sentinel/
├── index.html            # Main web application UI (Glassmorphic Interface)
├── vercel.json           # Vercel serverless deployment configuration
├── requirements.txt      # Python dependencies
├── .gitignore            # Git exclusions
├── api/
│   └── index.py          # Serverless Python handler for cloud deployments
├── static/
│   ├── samples/          # Test gallery insulator captures
│   └── output_*.jpg      # Historical detection previews
├── frontend/
│   └── index.html        # Modular frontend build
└── backend/
    ├── best.pt           # Trained YOLO model weights
    ├── main.py           # FastAPI backend server with live directory watcher
    ├── requirements.txt  # Backend dependencies
    └── uploads/          # Live drone footage ingestion folder
```

---

## 💻 Quick Start & Local Development

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/aero-sentinel.git
cd aero-sentinel
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI AI Server
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

### 4. Open in Browser
Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🚀 1-Click Cloud Deployment (Vercel)

1. Push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Aero Sentinel SIH26201"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/aero-sentinel.git
   git push -u origin main
   ```
2. Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Import your `aero-sentinel` repository and click **"Deploy"**.

---

## 🏆 Project Details
- **Competition:** Smart India Hackathon 2026 (SIH 2026)
- **Team ID:** `SIH26201`
- **Theme:** Robotics & Drones
- **Problem Statement:** AI-Powered Smart Inspection System for High-Voltage Insulator Fault Detection
- **Model Architecture:** YOLOv11 trained on High-Voltage Insulator Dataset
