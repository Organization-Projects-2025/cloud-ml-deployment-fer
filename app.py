from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T
from collections import deque
import sys
from pathlib import Path
import tempfile
import os

BASE_DIR = Path(_file_).resolve().parent
sys.path.insert(0, str(BASE_DIR / "model"))
from ddam_networks.DDAM import DDAMNet


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------


FER7 = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Angry"]
INPUT_SIZE = 112
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

device = torch.device("cpu")

model = None
face_cascade = None
transform = None


class EmotionSmoother:
    def _init_(self, window_size=10):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, probabilities):
        self.history.append(probabilities)
        return np.mean(self.history, axis=0)

    def get_stable_emotion(self):
        if not self.history:
            return None, 0.0
        smoothed = np.mean(self.history, axis=0)
        emotion_idx = int(np.argmax(smoothed))
        return FER7[emotion_idx], float(smoothed[emotion_idx])


def make_inference_transform(image_size=112):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def load_ddamfn_model(model_path, device):
    net = DDAMNet(num_class=7, num_head=2, pretrained=False)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    net.load_state_dict(state_dict, strict=False)
    net.to(device)
    net.eval()
    return net


def unpack_model_output(out):
    if isinstance(out, dict):
        logits = out.get("expr", out.get("logits", out))
    elif isinstance(out, (list, tuple)):
        logits = None
        for el in out:
            if torch.is_tensor(el) and el.dim() == 2 and el.size(1) >= 7:
                logits = el
                break
        if logits is None:
            logits = out[-1] if torch.is_tensor(out[-1]) else out[0]
    else:
        logits = out
    return logits, None


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))


def predict_emotion(face_img, net, device):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = net(face_tensor)
        logits, _ = unpack_model_output(outputs)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

    return FER7[predicted_class], confidence, probabilities


@app.on_event("startup")
async def startup_event():
    global model, face_cascade, transform

    model_path = BASE_DIR / "model" / "fer_model.pth"
    model = load_ddamfn_model(str(model_path), device)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    transform = make_inference_transform(INPUT_SIZE)


@app.get("/healthz")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "device": str(device)}


@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Only video files supported")

    contents = await file.read()
    tmp_path = None
    cap = None

    try:
        suffix = Path(file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open uploaded video")

        results = []
        frame_count = 0
        smoother = EmotionSmoother(window_size=15)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces(frame)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]

                padding = int(0.2 * w)
                x1, y1 = max(0, x - padding), max(0, y - padding)
                x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)

                face_img = frame[y1:y2, x1:x2]
                if face_img.shape[0] >= 48 and face_img.shape[1] >= 48:
                    emotion, confidence, probs = predict_emotion(face_img, model, device)
                    smoother.update(probs)
                    stable_emotion, stable_conf = smoother.get_stable_emotion()

                    results.append({
                        "frame": frame_count,
                        "face_bbox": [int(x), int(y), int(w), int(h)],
                        "emotion": stable_emotion,
                        "confidence": float(stable_conf),
                        "raw_emotion": emotion,
                        "raw_confidence": float(confidence),
                    })

            frame_count += 1

        return {
            "total_frames": frame_count,
            "faces_detected": len(results),
            "predictions": results[-10:] if len(results) > 10 else results
        }

    finally:
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if _name_ == "_main_":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)