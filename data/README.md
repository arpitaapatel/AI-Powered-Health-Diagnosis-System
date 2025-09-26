# Data Setup

This repository does not include raw datasets. Please download one or both of the following Kaggle datasets and organize them into class folders:

- Chest X-Ray Pneumonia (`https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia`)
- COVID-19 Radiography Database (`https://www.kaggle.com/tawsifurrahman/covid19-radiography-database`)

Recommended unified layout (merge if using both):

```
data/
  Normal/
  Pneumonia/
  COVID-19/
```

Place chest X-ray images (PNG/JPG) into the corresponding class folders. The training script will create train/val/test splits automatically.

Example training command:

```bash
python -m src.medimg.train --data_dir /absolute/path/to/data --output_dir artifacts --epochs 15 --batch_size 32 --image_size 224 224
```

After training, the best model alias and sidecar class names are saved under `models/best_model.keras` and `models/best_model_class_names.json`.


