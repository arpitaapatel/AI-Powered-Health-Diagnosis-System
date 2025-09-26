import os
import json
import shutil
from typing import Tuple, List

import tensorflow as tf
import numpy as np

from .data import create_datasets
from .models import build_scratch_cnn, build_resnet50, build_efficientnetb0
from .utils import compile_model, evaluate_model, plot_confusion_matrix, plot_roc_curve


def train_and_evaluate(
    data_dir: str,
    output_dir: str = "artifacts",
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 10,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    train_ds, val_ds, test_ds, class_names = create_datasets(data_dir, image_size=image_size, batch_size=batch_size)

    num_classes = len(class_names)
    input_shape = (image_size[0], image_size[1], 3)

    models = {
        "scratch_cnn": build_scratch_cnn(input_shape, num_classes),
        "resnet50": build_resnet50(input_shape, num_classes, train_base=False),
        "efficientnetb0": build_efficientnetb0(input_shape, num_classes, train_base=False),
    }

    history_dict = {}
    results = {}
    best_model_name = None
    best_model_accuracy = -1.0

    for name, model in models.items():
        model = compile_model(model, num_classes)
        ckpt_path = os.path.join(output_dir, f"{name}.weights.keras")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_accuracy', mode='max'),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', mode='max'),
        ]
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        history_dict[name] = history.history

        eval_res = evaluate_model(model, test_ds, class_names)
        results[name] = eval_res

        # Track best by accuracy
        try:
            current_acc = float(eval_res["classification_report"]["accuracy"])
        except Exception:
            current_acc = 0.0
        if current_acc > best_model_accuracy:
            best_model_accuracy = current_acc
            best_model_name = name

        # Save metrics and plots
        np.save(os.path.join(output_dir, f"{name}_history.npy"), history.history, allow_pickle=True)
        np.save(os.path.join(output_dir, f"{name}_results.npy"), eval_res, allow_pickle=True)

        cm_fig = plot_confusion_matrix(eval_res["confusion_matrix"], class_names)
        cm_fig.savefig(os.path.join(output_dir, f"{name}_confusion_matrix.png"), dpi=150, bbox_inches='tight')
        if eval_res["fpr"] is not None:
            roc_fig = plot_roc_curve(eval_res["fpr"], eval_res["tpr"], eval_res["roc_auc"]) 
            roc_fig.savefig(os.path.join(output_dir, f"{name}_roc_curve.png"), dpi=150, bbox_inches='tight')

        # Save full model and class names sidecar in artifacts
        model_path = os.path.join(output_dir, f"{name}.keras")
        model.save(model_path)
        class_sidecar = os.path.join(output_dir, f"{name}_class_names.json")
        with open(class_sidecar, "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)

        # Also save a copy under models/ for convenience
        shutil.copyfile(model_path, os.path.join(models_dir, f"{name}.keras"))
        shutil.copyfile(class_sidecar, os.path.join(models_dir, f"{name}_class_names.json"))

    # Optional: fine-tune transfer models base
    for name in ["resnet50", "efficientnetb0"]:
        base_path = os.path.join(output_dir, f"{name}.keras")
        if os.path.exists(base_path):
            model = tf.keras.models.load_model(base_path)
            # Unfreeze some layers of base
            trainable_layers = 50  # heuristic small amount
            train_count = 0
            for layer in model.layers[::-1]:
                if train_count < trainable_layers:
                    layer.trainable = True
                    train_count += 1
                else:
                    break
            model = compile_model(model, num_classes, lr=1e-5)
            history = model.fit(train_ds, validation_data=val_ds, epochs=max(3, epochs//3))
            eval_res = evaluate_model(model, test_ds, class_names)
            finetuned_path = os.path.join(output_dir, f"{name}_finetuned.keras")
            model.save(finetuned_path)
            # Save sidecar for class names
            with open(os.path.join(output_dir, f"{name}_finetuned_class_names.json"), "w", encoding="utf-8") as f:
                json.dump(class_names, f, ensure_ascii=False, indent=2)
            # Mirror to models/
            shutil.copyfile(finetuned_path, os.path.join(models_dir, f"{name}_finetuned.keras"))
            shutil.copyfile(os.path.join(output_dir, f"{name}_finetuned_class_names.json"), os.path.join(models_dir, f"{name}_finetuned_class_names.json"))
            # Update best tracker if finetuning improved
            try:
                ft_acc = float(eval_res["classification_report"]["accuracy"])
                if ft_acc > best_model_accuracy:
                    best_model_accuracy = ft_acc
                    best_model_name = f"{name}_finetuned"
            except Exception:
                pass

    # Save a canonical best model alias under models/
    if best_model_name is not None:
        src_model = os.path.join(output_dir, f"{best_model_name}.keras")
        if os.path.exists(src_model):
            shutil.copyfile(src_model, os.path.join(models_dir, "best_model.keras"))
        src_sidecar = os.path.join(output_dir, f"{best_model_name}_class_names.json")
        if os.path.exists(src_sidecar):
            shutil.copyfile(src_sidecar, os.path.join(models_dir, "best_model_class_names.json"))
            np.save(os.path.join(output_dir, f"{name}_finetuned_history.npy"), history.history, allow_pickle=True)
            np.save(os.path.join(output_dir, f"{name}_finetuned_results.npy"), eval_res, allow_pickle=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate medical imaging classifiers.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to images directory (folders per class)")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Directory to store models and results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=(224, 224))
    args = parser.parse_args()

    train_and_evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


