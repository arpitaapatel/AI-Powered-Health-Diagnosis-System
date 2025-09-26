import os
from typing import Tuple, Optional, List

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def build_augmenter(image_size: Tuple[int, int]) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")


def _normalize(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def _preprocess(image: tf.Tensor, label: tf.Tensor, image_size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    image = _normalize(image)
    return image, label


def _preprocess_with_augment(image: tf.Tensor, label: tf.Tensor, image_size: Tuple[int, int], augmenter: tf.keras.Sequential) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=True)
    image = augmenter(image, training=True)
    image = _normalize(image)
    return image, label


def create_datasets(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 1337,
    class_names: Optional[List[str]] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if class_names is None:
        tmp = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=val_split + test_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
        )
        class_names = tmp.class_names
        del tmp

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split + test_split,
        subset="training",
        interpolation="bilinear",
    )

    valtest_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split + test_split,
        subset="validation",
        interpolation="bilinear",
    )

    val_size = int(len(valtest_ds) * (val_split / (val_split + test_split))) if (val_split + test_split) > 0 else 0
    val_ds = valtest_ds.take(val_size)
    test_ds = valtest_ds.skip(val_size)

    augmenter = build_augmenter(image_size)

    train_ds = train_ds.map(lambda x, y: _preprocess_with_augment(x, y, image_size, augmenter), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: _preprocess(x, y, image_size), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: _preprocess(x, y, image_size), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


