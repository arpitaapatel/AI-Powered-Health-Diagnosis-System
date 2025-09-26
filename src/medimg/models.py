from typing import Tuple, Optional

import tensorflow as tf


def build_scratch_cnn(input_shape: Tuple[int, int, int], num_classes: int, dropout_rate: float = 0.4) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="scratch_cnn")
    return model


def build_resnet50(input_shape: Tuple[int, int, int], num_classes: int, train_base: bool = False) -> tf.keras.Model:
    base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = train_base
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="resnet50_transfer")
    return model


def build_efficientnetb0(input_shape: Tuple[int, int, int], num_classes: int, train_base: bool = False) -> tf.keras.Model:
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = train_base
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_transfer")
    return model


