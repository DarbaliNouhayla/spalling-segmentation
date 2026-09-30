# Spalling Segmentation

A Streamlit web app for testing a SegFormer (MiT-b2) model trained to segment
concrete spalling in images. Upload an image and get the predicted spalling
mask, an overlay, and the affected area as a percentage.

## Live demo

[https://spalling-segmentation.streamlit.app/](https://spalling-segmentation.streamlit.app/)

## Model

- **Architecture:** SegFormer, MiT-b2 encoder (`segmentation_models_pytorch`)
- **Task:** binary semantic segmentation (spalling vs. background)
- **Input size:** 512×512, ImageNet normalization
- **Weights:** hosted on the Hugging Face Hub —
  [darbalinouhayla/spalling_segmentation_segformer_mitb2](https://huggingface.co/darbalinouhayla/spalling_segmentation_segformer_mitb2)
  (downloaded automatically at app startup, not stored in this repo)

## Running locally

```bash
python -m venv venv
venv\Scripts\activate        # on Windows
pip install -r requirements.txt
streamlit run app.py
```

The app will download the model checkpoint from Hugging Face on first run.

## Files

| File               | Purpose                                      |
|--------------------|-----------------------------------------------|
| `app.py`           | Streamlit app: UI, preprocessing, inference   |
| `requirements.txt` | Python dependencies                           |

## Notes

- Threshold and overlay opacity are adjustable from the sidebar.
- Predicted masks can be downloaded as PNG files.
- CPU inference takes roughly 1–3 seconds per image; no GPU required.
