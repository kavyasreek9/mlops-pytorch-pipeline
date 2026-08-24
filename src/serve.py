import io
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src. model import get_model

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

app = FastAPI(title="MLOps PyTorch Serving")

_state = {"model": None, "device": None}

_preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    architecture = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", 10)

    model = get_model(architecture=architecture, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _state["model"] = model
    _state["device"] = device


@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except FileNotFoundError:
        # Allow the app to boot even without a checkpoint yet; /health will
        # report not-ready and /predict will 503 until a checkpoint appears.
        _state["model"] = None


@app.get("/health")
def health():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    tensor = _preprocess(img).unsqueeze(0).to(_state["device"])

    with torch.no_grad():
        logits = _state["model"](tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    result = {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)}
    predicted_class = CLASS_NAMES[int(torch.tensor(probs).argmax())]

    return {"predicted_class": predicted_class, "probabilities": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
