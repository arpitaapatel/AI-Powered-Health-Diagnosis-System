# Medical Imaging AI: Chest X-ray Pneumonia/COVID-19 Detection

End-to-end data science project to detect Normal, Pneumonia, and COVID-19 from chest X-ray images.

## Features
- Training pipeline: CNN-from-scratch + transfer learning (ResNet50, EfficientNetB0)
- Preprocessing: resize 224x224, normalization, augmentation (flips, rotations, zoom, contrast)
- Evaluation: Accuracy, Precision, Recall, F1 (via report), ROC-AUC, Confusion Matrix and ROC plots
- Explainability: Grad-CAM heatmaps
- Deployment: Streamlit app for image upload, prediction, confidence, and Grad-CAM visualization

## Dataset
Use one or both of:
- Kaggle Chest X-Ray Pneumonia (`https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia`)
- COVID-19 Radiography Database (`https://www.kaggle.com/tawsifurrahman/covid19-radiography-database`)

Organize images into a single folder with subfolders per class, for example:
```
data/
  Normal/
  Pneumonia/
  COVID-19/
```
You may merge datasets by copying images into these class folders. The loader will split into train/val/test. See `data/README.md` for detailed instructions.

## Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, TensorFlow acceleration is enabled via `tensorflow-metal`.

## Train
```bash
python -m src.medimg.train --data_dir /path/to/data --output_dir artifacts --epochs 15 --batch_size 32 --image_size 224 224
```
Artifacts saved to `artifacts/`:
- Model weights: `scratch_cnn.keras`, `resnet50.keras`, `efficientnetb0.keras` and fine-tuned variants
- Training history and evaluation results (`*.npy`)
- Confusion matrices and ROC curves (`*.png`)

Additionally, convenience copies are saved under `models/`, including `best_model.keras` and `best_model_class_names.json` which point to the best-performing model by test accuracy.

## Streamlit App
```bash
streamlit run src/app/medimg_app.py --server.address=0.0.0.0 --server.port=8501
```
Ensure the `artifacts/` or `models/` directory with trained models exists. Select a model in the sidebar and upload an X-ray. The app automatically loads class names from sidecar JSON when available.

## Grad-CAM
Implemented in `src/medimg/utils.py`. The app overlays heatmaps automatically after prediction.

## Jupyter Notebook
An example notebook will be provided in `notebooks/medimg_e2e.ipynb` to demonstrate E2E training and inference.

## Deployment
- Streamlit Cloud: push repo and set app entrypoint `src/app/medimg_app.py`
- Heroku: use `Procfile` with `web: streamlit run src/app/medimg_app.py` and add buildpacks for Python

## Project Structure
```
data/
  README.md          # dataset instructions
models/              # saved models (populated after training)
notebooks/
  medimg_e2e.ipynb   # optional exploration
src/
  medimg/
    __init__.py
    data.py          # loaders with preprocessing + augmentation
    models.py        # scratch CNN + ResNet50/EfficientNet
    train.py         # training/eval, metrics, plots, saving models
    utils.py         # metrics, plotting, Grad-CAM
  app/
    medimg_app.py    # Streamlit web app
    streamlit_app.py # convenience entrypoint
```

## Screenshots
Add example predictions with Grad-CAM overlays here (e.g., `artifacts/*_confusion_matrix.png`).

## Notes
- This project is for educational and research use only and is not a medical device.