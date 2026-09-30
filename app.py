import io

import numpy as np
import segmentation_models_pytorch as smp
import streamlit as st
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

# ---- must match your training script ----
HF_REPO_ID = "darbalinouhayla/spalling_segmentation_segformer_mitb2"
HF_FILENAME = "best_model.pth"
IMG_SIZE = 512            # CONFIG["image_size"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # IMAGENET_MEAN
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)    # IMAGENET_STD
DEFAULT_THRESHOLD = 0.5   # CONFIG["segmentation_threshold"]
# -----------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Spalling Segmentation", layout="wide")


@st.cache_resource(show_spinner="Downloading and loading model...")
def load_model():
    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    encoder = cfg.get("encoder_name", "mit_b2")

    # Same constructor as training: 3 input channels, 1 output channel (sigmoid).
    # encoder_weights=None -> no ImageNet download; our weights come from the checkpoint.
    model = smp.Segformer(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)  # fails loudly on any mismatch

    info = {
        "encoder": encoder,
        "epoch": ckpt.get("epoch"),
        "val_positive_iou": ckpt.get("val_positive_iou"),
    }
    return model.to(DEVICE).eval(), info


@torch.inference_mode()
def get_probs(model, image):
    h, w = image.shape[:2]

    resized = np.array(
        Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    resized = (resized - MEAN) / STD
    x = torch.from_numpy(resized).permute(2, 0, 1)[None].to(DEVICE)

    logits = model(x)  # (1, 1, 512, 512)
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


def make_overlay(image, mask, alpha):
    overlay = image.astype(np.float32).copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array([255, 0, 0])
    return overlay.astype(np.uint8)


def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ------------------------- UI -------------------------
st.title("Spalling Segmentation (SegFormer MiT-b2)")

try:
    model, info = load_model()
except Exception as e:
    st.error(f"Could not load the checkpoint: {e}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Probability threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05)
    alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)
    st.caption(f"Device: {DEVICE}")
    st.caption(f"Encoder: {info['encoder']}")
    if info["epoch"] is not None:
        st.caption(f"Checkpoint epoch: {info['epoch']}")
    if info["val_positive_iou"] is not None:
        st.caption(f"Val positive IoU: {info['val_positive_iou']:.4f}")

files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
)

for f in files or []:
    image = np.array(Image.open(f).convert("RGB"))
    probs = get_probs(model, image)
    mask = probs >= threshold

    st.subheader(f.name)
    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="Original", use_container_width=True)
    c2.image(make_overlay(image, mask, alpha), caption="Overlay", use_container_width=True)
    c3.image((mask * 255).astype(np.uint8), caption="Binary mask", use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Spalling area", f"{mask.mean() * 100:.2f}%")
    m2.metric("Max probability", f"{probs.max():.3f}")

    st.download_button(
        "Download mask (PNG)",
        to_png_bytes((mask * 255).astype(np.uint8)),
        file_name=f"{f.name.rsplit('.', 1)[0]}_mask.png",
        mime="image/png",
        key=f"dl_{f.name}",
    )
    st.divider()

if not files:
    st.info("Upload an image to run the model.")