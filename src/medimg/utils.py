from typing import Dict, Tuple, List, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt


def compile_model(model: tf.keras.Model, num_classes: int, lr: float = 1e-4) -> tf.keras.Model:
    loss = "sparse_categorical_crossentropy"
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc", multi_label=False, num_labels=None),
    ]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)
    return model


def evaluate_model(model: tf.keras.Model, ds: tf.data.Dataset, class_names: List[str]) -> Dict:
    y_true = []
    y_prob = []
    for batch_x, batch_y in ds:
        preds = model.predict(batch_x, verbose=0)
        y_true.append(batch_y.numpy())
        y_prob.append(preds)
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            fpr, tpr = None, None
    except Exception:
        auc, fpr, tpr = None, None, None

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], figsize: Tuple[int, int] = (5, 4)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=class_names, yticklabels=class_names, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_roc_curve(fpr: Optional[np.ndarray], tpr: Optional[np.ndarray], auc: Optional[float]) -> Optional[plt.Figure]:
    if fpr is None or tpr is None:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate', title='Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def compute_gradcam_heatmap(model: tf.keras.Model, image: tf.Tensor, class_index: Optional[int] = None, last_conv_layer_name: Optional[str] = None) -> np.ndarray:
    if last_conv_layer_name is None:
        # Heuristic: pick last conv layer
        conv_layers = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            # Try to inspect model layers for conv in nested models
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    conv_layers.extend([l.name for l in layer.layers if isinstance(l, tf.keras.layers.Conv2D)])
        if not conv_layers:
            raise ValueError("No Conv2D layers found for Grad-CAM.")
        last_conv_layer_name = conv_layers[-1]

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.cast(image, tf.float32))
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35, cmap: str = 'jet') -> np.ndarray:
    import cv2

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, getattr(cv2, f'COLORMAP_{cmap.upper()}', cv2.COLORMAP_JET))
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored = cv2.resize(colored, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(colored, alpha, image, 1 - alpha, 0)
    return overlay


