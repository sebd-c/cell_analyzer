# imports
import argparse
import csv
import math
import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, LayerNormalization,
                                     GlobalAveragePooling1D, MultiHeadAttention,
                                     Embedding, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from src.models.image_based.dataloader.dataset_api import (
    build_stratified_kfold_benchmark,
    build_stratified_benchmark,
    check_paired_consistency,
)

#################################################################################

def build_model(img_size, num_channels, num_classes, lr):
    """Build the ResNet50 classifier (kept for backwards-compatible imports)."""
    weights = "imagenet" if num_channels == 3 else None
    if weights is None:
        print("Warning: num_channels != 3, using random initialization for ResNet50.")

    base_model = ResNet50(weights=weights,
                          include_top=False,
                          input_shape=(*img_size, num_channels)
                          )

    base_model.trainable = False  # transfer learning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes,
                    activation="softmax",
                    kernel_regularizer=l2(1e-4))(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.backbone = base_model

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"]
                  )

    return model


def build_convnext_model(img_size, num_channels, num_classes, lr):
    """Build a ConvNeXtBase classifier with an ImageNet-initialized backbone."""
    if num_channels != 3:
        raise ValueError("ConvNeXtBase requires three input channels.")

    try:
        from tensorflow.keras.applications import ConvNeXtBase
    except ImportError as exc:
        raise ImportError(
            "ConvNeXt requires a TensorFlow/Keras version that provides "
            "tf.keras.applications.ConvNeXtBase."
        ) from exc

    # ConvNeXt includes its own [0, 255] -> normalized preprocessing layer.
    base_model = ConvNeXtBase(weights="imagenet",
                             include_top=False,
                             input_shape=(*img_size, num_channels),
                             include_preprocessing=True)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes,
                    activation="softmax",
                    kernel_regularizer=l2(1e-4))(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.backbone = base_model
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_vgg16_model(img_size, num_channels, num_classes, lr):
    """Build an ImageNet-pretrained VGG16 transfer-learning classifier."""
    if num_channels != 3:
        raise ValueError("VGG16 requires three input channels.")

    inputs = tf.keras.Input(shape=(*img_size, num_channels))
    backbone = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, num_channels),
    )
    backbone.trainable = False

    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=l2(1e-4),
    )(x)
    model = Model(inputs=inputs, outputs=outputs, name="vgg16_classifier")
    model.backbone = backbone
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


VIT_PRETRAINED_CHECKPOINT = (
    "gs://vit_models/augreg/"
    "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz"
)


@tf.keras.utils.register_keras_serializable(package="cell_analyzer")
class _ViTClassToken(tf.keras.layers.Layer):
    """Learned [CLS] token used by the original Google ViT implementation."""

    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.token = self.add_weight(
            name="cls",
            shape=(1, 1, self.hidden_size),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.broadcast_to(
            self.token,
            [tf.shape(inputs)[0], 1, self.hidden_size],
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size})
        return config


@tf.keras.utils.register_keras_serializable(package="cell_analyzer")
class _ViTAttention(tf.keras.layers.Layer):
    """Multi-head self-attention with weight shapes matching Flax ViT."""

    def __init__(self, hidden_size, num_heads, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = Dense(hidden_size, name="query")
        self.key = Dense(hidden_size, name="key")
        self.value = Dense(hidden_size, name="value")
        # `output` is a reserved read-only property on Keras layers.
        self.output_projection = Dense(hidden_size, name="out")
        self.dropout = Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]

        def split_heads(x):
            x = tf.reshape(
                x,
                [batch_size, sequence_length, self.num_heads, self.head_size],
            )
            return tf.transpose(x, [0, 2, 1, 3])

        query = split_heads(self.query(inputs))
        key = split_heads(self.key(inputs))
        value = split_heads(self.value(inputs))

        attention = tf.matmul(query, key, transpose_b=True)
        attention *= tf.math.rsqrt(tf.cast(self.head_size, tf.float32))
        attention = tf.nn.softmax(attention, axis=-1)
        attention = self.dropout(attention, training=training)
        output = tf.matmul(attention, value)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(
            output,
            [batch_size, sequence_length, self.hidden_size],
        )
        return self.output_projection(output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="cell_analyzer")
class _ViTEncoderBlock(tf.keras.layers.Layer):
    """Pre-norm Transformer block used by the released ViT-B/16 checkpoint."""

    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.norm1 = LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.attention = _ViTAttention(
            hidden_size,
            num_heads,
            dropout_rate=0.0,
            name="MultiHeadDotProductAttention_1",
        )
        self.attention_dropout = Dropout(dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6, name="LayerNorm_1")
        self.mlp_dense_0 = Dense(mlp_dim, activation=tf.nn.gelu, name="Dense_0")
        self.mlp_dropout_0 = Dropout(dropout_rate)
        self.mlp_dense_1 = Dense(hidden_size, name="Dense_1")
        self.mlp_dropout_1 = Dropout(dropout_rate)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

    def call(self, inputs, training=False):
        attention = self.attention(self.norm1(inputs), training=training)
        x = inputs + self.attention_dropout(attention, training=training)
        mlp = self.mlp_dense_0(self.norm2(x))
        mlp = self.mlp_dropout_0(mlp, training=training)
        mlp = self.mlp_dense_1(mlp)
        mlp = self.mlp_dropout_1(mlp, training=training)
        return x + mlp

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="cell_analyzer")
class _ViTB16Backbone(tf.keras.Model):
    """Keras implementation of Google's ViT-B/16 encoder."""

    def __init__(self, img_size, hidden_size=768, num_heads=12,
                 transformer_layers=12, mlp_dim=3072, **kwargs):
        kwargs.setdefault("name", "vit_b16_backbone")
        super().__init__(**kwargs)
        height, width = img_size
        if height % 16 or width % 16:
            raise ValueError("ViT-B/16 requires image dimensions divisible by 16.")
        self.height = height
        self.width = width
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.mlp_dim = mlp_dim
        self.patch_embedding = tf.keras.layers.Conv2D(
            hidden_size,
            kernel_size=16,
            strides=16,
            padding="valid",
            name="embedding",
        )
        self.class_token = _ViTClassToken(hidden_size, name="cls")
        num_tokens = (height // 16) * (width // 16) + 1
        self.position_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, num_tokens, hidden_size),
            initializer="zeros",
            trainable=True,
        )
        self.position_dropout = Dropout(0.1)
        self.encoder_blocks = [
            _ViTEncoderBlock(
                hidden_size,
                num_heads,
                mlp_dim,
                dropout_rate=0.1,
                name=f"encoderblock_{index}",
            )
            for index in range(transformer_layers)
        ]
        self.encoder_norm = LayerNormalization(
            epsilon=1e-6,
            name="encoder_norm",
        )

    def call(self, inputs, training=False):
        patches = self.patch_embedding(inputs)
        tokens = tf.reshape(
            patches,
            [tf.shape(patches)[0], -1, self.hidden_size],
        )
        tokens = tf.concat([self.class_token(tokens), tokens], axis=1)
        tokens = tokens + self.position_embedding
        tokens = self.position_dropout(tokens, training=training)
        for block in self.encoder_blocks:
            tokens = block(tokens, training=training)
        tokens = self.encoder_norm(tokens)
        return tokens[:, 0]

    def get_config(self):
        config = super().get_config()
        config.update({
            "img_size": (self.height, self.width),
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "transformer_layers": self.transformer_layers,
            "mlp_dim": self.mlp_dim,
        })
        return config


def _load_vit_npz(uri):
    """Load a public Google Cloud ViT checkpoint without requiring JAX/Flax."""
    try:
        with tf.io.gfile.GFile(uri, "rb") as checkpoint_file:
            with np.load(checkpoint_file, allow_pickle=False) as checkpoint:
                return {key: checkpoint[key] for key in checkpoint.files}
    except Exception as exc:
        raise RuntimeError(
            f"Could not read ViT checkpoint {uri!r}. The public GCS object must "
            "be reachable from this environment."
        ) from exc


def _vit_parameter(parameters, key):
    try:
        return parameters[key]
    except KeyError as exc:
        raise KeyError(f"Missing parameter {key!r} in the ViT checkpoint.") from exc


def _set_layer_norm_weights(layer, parameters, prefix):
    layer.set_weights([
        _vit_parameter(parameters, f"{prefix}/scale"),
        _vit_parameter(parameters, f"{prefix}/bias"),
    ])


def _set_attention_weights(attention, parameters, prefix):
    for dense, name in (
        (attention.query, "query"),
        (attention.key, "key"),
        (attention.value, "value"),
    ):
        kernel = _vit_parameter(parameters, f"{prefix}/{name}/kernel")
        bias = _vit_parameter(parameters, f"{prefix}/{name}/bias")
        dense.set_weights([kernel.reshape(attention.hidden_size, -1), bias.reshape(-1)])

    output_kernel = _vit_parameter(parameters, f"{prefix}/out/kernel")
    output_bias = _vit_parameter(parameters, f"{prefix}/out/bias")
    attention.output_projection.set_weights([
        output_kernel.reshape(attention.hidden_size, attention.hidden_size),
        output_bias,
    ])


def _load_vit_b16_weights(backbone, parameters):
    embedding = _vit_parameter(parameters, "embedding/kernel")
    embedding_bias = _vit_parameter(parameters, "embedding/bias")
    backbone.patch_embedding.set_weights([embedding, embedding_bias])

    pretrained_cls = _vit_parameter(parameters, "cls")
    backbone.class_token.token.assign(pretrained_cls)

    pretrained_position = _vit_parameter(
        parameters,
        "Transformer/posembed_input/pos_embedding",
    )
    target_tokens = backbone.position_embedding.shape[1]
    if pretrained_position.shape[1] != target_tokens:
        class_position = pretrained_position[:, :1]
        grid_position = pretrained_position[:, 1:]
        old_grid_size = int(math.sqrt(grid_position.shape[1]))
        new_grid_size = int(math.sqrt(target_tokens - 1))
        if old_grid_size ** 2 != grid_position.shape[1]:
            raise ValueError("ViT checkpoint positional embedding is not square.")
        if new_grid_size ** 2 != target_tokens - 1:
            raise ValueError("Target ViT image size produces a non-square patch grid.")
        grid_position = grid_position.reshape(
            1, old_grid_size, old_grid_size, backbone.hidden_size
        )
        grid_position = tf.image.resize(
            grid_position,
            [new_grid_size, new_grid_size],
            method="bicubic",
        ).numpy()
        pretrained_position = np.concatenate([class_position, grid_position.reshape(
            1, new_grid_size * new_grid_size, backbone.hidden_size
        )], axis=1)
    backbone.position_embedding.assign(pretrained_position)

    for index, block in enumerate(backbone.encoder_blocks):
        prefix = f"Transformer/encoderblock_{index}"
        _set_layer_norm_weights(block.norm1, parameters, f"{prefix}/LayerNorm_0")
        _set_attention_weights(
            block.attention,
            parameters,
            f"{prefix}/MultiHeadDotProductAttention_1",
        )
        # Flax's module auto-naming reserves LayerNorm_1 for the attention
        # submodule, so the second encoder-block norm is LayerNorm_2.
        _set_layer_norm_weights(block.norm2, parameters, f"{prefix}/LayerNorm_2")
        block.mlp_dense_0.set_weights([
            _vit_parameter(parameters, f"{prefix}/MlpBlock_3/Dense_0/kernel"),
            _vit_parameter(parameters, f"{prefix}/MlpBlock_3/Dense_0/bias"),
        ])
        block.mlp_dense_1.set_weights([
            _vit_parameter(parameters, f"{prefix}/MlpBlock_3/Dense_1/kernel"),
            _vit_parameter(parameters, f"{prefix}/MlpBlock_3/Dense_1/bias"),
        ])

    _set_layer_norm_weights(
        backbone.encoder_norm,
        parameters,
        "Transformer/encoder_norm",
    )


def build_vit_model(img_size, num_channels, num_classes, lr, patch_size=16,
                    projection_dim=768, num_heads=12, transformer_layers=12,
                    mlp_dim=3072):
    """Build ViT-B/16 and load Google's AugReg ImageNet-21k checkpoint."""
    height, width = img_size
    if num_channels != 3:
        raise ValueError("Pretrained ViT-B/16 requires three input channels.")
    expected = (16, 768, 12, 12, 3072)
    supplied = (patch_size, projection_dim, num_heads, transformer_layers, mlp_dim)
    if supplied != expected:
        raise ValueError(
            "The selected checkpoint is fixed to ViT-B/16 architecture: "
            "patch_size=16, projection_dim=768, num_heads=12, "
            "transformer_layers=12, mlp_dim=3072."
        )

    inputs = tf.keras.Input(shape=(height, width, num_channels))
    backbone = _ViTB16Backbone(img_size=img_size)
    features = backbone(inputs)
    x = Dropout(0.2, name="classifier_dropout")(features)
    outputs = Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=l2(1e-4),
        name="classifier",
    )(x)
    model = Model(inputs=inputs, outputs=outputs, name="vision_transformer_l16")
    _ = model(tf.zeros((1, height, width, num_channels)))
    _load_vit_b16_weights(backbone, _load_vit_npz(VIT_PRETRAINED_CHECKPOINT))
    backbone.trainable = False
    model.backbone = backbone
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_custom_cnn(img_size, num_channels, num_classes, lr):
    """Build an EfficientNetB0 transfer-learning CNN classifier."""
    if num_channels != 3:
        raise ValueError("Pretrained CNN requires three input channels.")

    inputs = tf.keras.Input(shape=(*img_size, num_channels))
    backbone = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, num_channels),
    )
    backbone.trainable = False
    x = backbone(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes,
                    activation="softmax",
                    kernel_regularizer=l2(1e-4))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.backbone = backbone
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"]
                  )
    return model

def _compute_balanced_metrics(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall),
                      0.0)

    balanced_accuracy = float(np.mean(recall))
    balanced_precision = float(np.mean(precision))
    balanced_f1 = float(np.mean(f1))

    return balanced_accuracy, balanced_precision, balanced_f1


def _unfreeze_backbone_tail(model, fraction, learning_rate):
    """Unfreeze the tail of a pretrained backbone and recompile the model.

    ViT stores its Transformer blocks in a tracked Python list. Using the
    backbone's top-level ``layers`` list would otherwise turn the default 5%
    into just the final LayerNorm, leaving every Transformer block frozen.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return False

    if not 0.0 < fraction <= 1.0:
        raise ValueError("unfreeze_fraction must be greater than 0 and at most 1.")

    backbone.trainable = True

    vit_blocks = getattr(backbone, "encoder_blocks", None)
    if vit_blocks is not None:
        # For ViT, define the fraction over Transformer blocks rather than
        # over the backbone's top-level bookkeeping layers.
        n_unfrozen = max(1, math.ceil(len(vit_blocks) * fraction))
        for layer in backbone.layers:
            layer.trainable = False
        for block in vit_blocks[-n_unfrozen:]:
            block.trainable = True
        backbone.encoder_norm.trainable = True
    else:
        backbone_layers = list(backbone.layers)
        n_unfrozen = max(1, math.ceil(len(backbone_layers) * fraction))
        for layer in backbone_layers[:-n_unfrozen]:
            layer.trainable = False
        for layer in backbone_layers[-n_unfrozen:]:
            # BatchNorm statistics are unreliable with small fine-tuning
            # batches.
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return True


def _make_callbacks(out_dir, model_name, fold):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(out_dir, f"{model_name}_fold{fold}.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        _EpochLossPrinter(),
    ]


class _EpochLossPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        if loss is not None:
            if val_loss is not None:
                print(f"Epoch {epoch + 1}: loss={loss:.4f} val_loss={val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}: loss={loss:.4f}")


def _evaluate_dataset(model, dataset, num_classes):
    y_true = []
    y_pred = []
    for batch_x, batch_y in dataset:
        preds = model(batch_x, training=False)
        y_true.append(batch_y.numpy())
        y_pred.append(tf.argmax(preds, axis=1).numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    bal_metrics = _compute_balanced_metrics(y_true, y_pred, num_classes)
    return bal_metrics, cm


def _save_evaluation_csv(out_dir, model_name, fold, evaluations, class_names):
    """Save scalar evaluation metrics in a tidy CSV file."""
    output_path = os.path.join(
        out_dir,
        f"{model_name}_fold{fold}_evaluation.csv",
    )

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "record_type",
                "metric",
                "true_class",
                "predicted_class",
                "value",
            ],
        )
        writer.writeheader()

        metric_names = (
            "balanced_accuracy",
            "balanced_precision",
            "balanced_f1",
        )
        for split, (metrics, _) in evaluations.items():
            for metric_name, value in zip(metric_names, metrics):
                writer.writerow({
                    "split": split,
                    "record_type": "metric",
                    "metric": metric_name,
                    "true_class": "",
                    "predicted_class": "",
                    "value": float(value),
                })

    print(f"Saved evaluation CSV: {output_path}")
    return output_path


def _save_confusion_matrix_plots(out_dir, model_name, fold, evaluations, class_names):
    """Save one annotated raw-count confusion-matrix image per split."""
    for split, (_, confusion_matrix) in evaluations.items():
        figure_size = max(6, len(class_names) * 1.2)
        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        image = ax.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted class",
            ylabel="True class",
            title=f"{model_name} fold {fold} - {split} confusion matrix",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        threshold = confusion_matrix.max() / 2.0 if confusion_matrix.size else 0
        for row in range(confusion_matrix.shape[0]):
            for column in range(confusion_matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    int(confusion_matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if confusion_matrix[row, column] > threshold else "black",
                )

        fig.tight_layout()
        output_path = os.path.join(
            out_dir,
            f"{model_name}_fold{fold}_{split}_confusion_matrix.png",
        )
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved confusion matrix image: {output_path}")


def _compute_class_weights(samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for sample in samples:
        label = sample[-1]
        counts[label] += 1

    total = counts.sum()
    weights = {}
    for i, c in enumerate(counts):
        if c == 0:
            weights[i] = 0.0
        else:
            weights[i] = float(total) / (num_classes * float(c))

    return counts, weights


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    if args.data_dir2:
        consistency = check_paired_consistency(args.data_dir,
                                               args.data_dir2,
                                               images_extension=args.images_extension)
        if consistency["only_in_a"] or consistency["only_in_b"]:
            print("Warning: class folders do not match across data_dir and data_dir2.")
            if consistency["only_in_a"]:
                print(f"  Only in data_dir: {consistency['only_in_a']}")
            if consistency["only_in_b"]:
                print(f"  Only in data_dir2: {consistency['only_in_b']}")
        if consistency["missing_in_b"] or consistency["missing_in_a"]:
            print("Warning: filename mismatches detected between paired roots.")
            for cls, files in list(consistency["missing_in_b"].items())[:3]:
                print(f"  Missing in data_dir2 ({cls}): {files[:5]}")
            for cls, files in list(consistency["missing_in_a"].items())[:3]:
                print(f"  Missing in data_dir ({cls}): {files[:5]}")
    if args.data_dir3 and not args.data_dir2:
        raise ValueError("--data_dir3 requires --data_dir2; provide all three channel roots.")
    if args.data_dir3:
        consistency = check_paired_consistency(
            args.data_dir,
            args.data_dir3,
            images_extension=args.images_extension,
        )
        if consistency["only_in_a"] or consistency["only_in_b"]:
            print("Warning: class folders do not match across the first and third channel roots.")
        if consistency["missing_in_b"] or consistency["missing_in_a"]:
            print("Warning: filename mismatches detected for the third channel root.")

    effective_channels = args.num_channels
    imagenet_models = {"resnet50", "convnext", "custom", "vgg16", "vit"}
    if args.model in imagenet_models and args.num_channels in (1, 2):
        print(f"Warning: {args.model} expects 3 channels; upconverting to 3 channels.")
        effective_channels = 3

    if args.mode == "kfold":
        benchmark = build_stratified_kfold_benchmark(data_dir=args.data_dir,
                                                     batch_size=args.batch_size,
                                                     k=args.num_folds,
                                                     img_size=(args.img_size, args.img_size),
                                                     num_channels=effective_channels,
                                                     data_dir2=args.data_dir2,
                                                     data_dir3=args.data_dir3,
                                                     images_extension=args.images_extension,
                                                     seed=args.seed
                                                     )

        fold_data = benchmark["folds"][args.fold]
        train_ds = fold_data["train"]
        val_ds = fold_data["val"]
        train_samples = fold_data["train_samples"]

        num_classes = benchmark["num_classes"]
        class_names = benchmark["class_names"]

        print(f"Fold {args.fold}/{args.num_folds - 1}")
        print(f"Classes ({num_classes}): {class_names}")
        test_ds = None
    else:
        benchmark = build_stratified_benchmark(data_dir=args.data_dir,
                                               batch_size=args.batch_size,
                                               img_size=(args.img_size, args.img_size),
                                               num_channels=effective_channels,
                                               data_dir2=args.data_dir2,
                                               data_dir3=args.data_dir3,
                                               images_extension=args.images_extension,
                                               seed=args.seed
                                               )
        train_ds = benchmark["train"]
        val_ds = benchmark["val"]
        test_ds = benchmark["test"]
        train_samples = benchmark["train_samples"]
        num_classes = benchmark["num_classes"]
        class_names = benchmark["class_names"]
        print(f"Classes ({num_classes}): {class_names}")

    if args.data_dir2 and args.num_channels != 2:
        if not args.data_dir3 or args.num_channels != 3:
            print("Warning: data_dir2 provided but num_channels is not 2 or 3.")
    if args.data_dir3 and args.num_channels != 3:
        print("Warning: data_dir3 provided but num_channels is not 3.")

    if args.model in imagenet_models:
        def imagenet_preprocess(x, y):
            # The dataset API emits float images in [0, 1]. ResNet, ConvNeXt,
            # and EfficientNet expect values expressed in [0, 255]. The
            # released Google ViT checkpoint was trained with [-1, 1] inputs.
            if args.model == "resnet50":
                x = preprocess_input(x * 255.0)
            elif args.model == "vgg16":
                x = vgg16_preprocess_input(x * 255.0)
            elif args.model == "vit":
                x = x * 2.0 - 1.0
            else:
                x = x * 255.0
            return x, y

        train_ds = train_ds.map(imagenet_preprocess,
                                num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(imagenet_preprocess,
                            num_parallel_calls=tf.data.AUTOTUNE)
        if test_ds is not None:
            test_ds = test_ds.map(imagenet_preprocess,
                                  num_parallel_calls=tf.data.AUTOTUNE)

    if args.model == "resnet50":
        model = build_model(img_size=(args.img_size, args.img_size),
                            num_channels=effective_channels,
                            num_classes=num_classes,
                            lr=args.lr
                            )
    elif args.model == "convnext":
        model = build_convnext_model(img_size=(args.img_size, args.img_size),
                                     num_channels=effective_channels,
                                     num_classes=num_classes,
                                     lr=args.lr)
    elif args.model == "vit":
        model = build_vit_model(img_size=(args.img_size, args.img_size),
                                num_channels=effective_channels,
                                num_classes=num_classes,
                                lr=args.lr,
                                patch_size=args.vit_patch_size,
                                projection_dim=args.vit_projection_dim,
                                num_heads=args.vit_num_heads,
                                transformer_layers=args.vit_transformer_layers,
                                mlp_dim=args.vit_mlp_dim)
    elif args.model == "vgg16":
        model = build_vgg16_model(img_size=(args.img_size, args.img_size),
                                  num_channels=effective_channels,
                                  num_classes=num_classes,
                                  lr=args.lr)
    else:
        model = build_custom_cnn(img_size=(args.img_size, args.img_size),
                                 num_channels=effective_channels,
                                 num_classes=num_classes,
                                 lr=args.lr
                                 )

    model.summary()

    class_weight = None
    if args.class_weight == "balanced":
        counts, weights = _compute_class_weights(train_samples, num_classes)
        print(f"Train class counts: {counts.tolist()}")
        print(f"Using balanced class weights: {weights}")
        class_weight = weights

    # Train a new head first, then fine-tune the tail of pretrained
    # ImageNet backbones using a much smaller learning rate.
    is_pretrained_backbone = args.model in {
        "resnet50", "convnext", "custom", "vgg16", "vit"
    }
    warmup_epochs = min(args.warmup_epochs, args.epochs)

    if is_pretrained_backbone and warmup_epochs > 0:
        print(f"Training frozen {args.model} backbone for {warmup_epochs} epochs.")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup_epochs,
            callbacks=_make_callbacks(args.out_dir, args.model, args.fold),
            class_weight=class_weight,
        )

    if is_pretrained_backbone and args.epochs > warmup_epochs:
        _unfreeze_backbone_tail(
            model,
            fraction=args.unfreeze_fraction,
            learning_rate=args.finetune_lr,
        )
        print(
            f"Fine-tuning the last {args.unfreeze_fraction:.0%} of the "
            f"{args.model} backbone at learning rate {args.finetune_lr:g}."
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=warmup_epochs,
            epochs=args.epochs,
            callbacks=_make_callbacks(args.out_dir, args.model, args.fold),
            class_weight=class_weight,
        )
    elif not is_pretrained_backbone:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=_make_callbacks(args.out_dir, args.model, args.fold),
            class_weight=class_weight,
        )

    train_metrics, train_cm = _evaluate_dataset(model, train_ds, num_classes)
    val_metrics, val_cm = _evaluate_dataset(model, val_ds, num_classes)
    evaluations = {
        "train": (train_metrics, train_cm),
        "validation": (val_metrics, val_cm),
    }
    bal_acc, bal_prec, bal_f1 = val_metrics
    print("Validation report:")
    print(f"  Balanced accuracy: {bal_acc:.4f}")
    print(f"  Balanced precision: {bal_prec:.4f}")
    print(f"  Balanced f1-score: {bal_f1:.4f}")
    print("Train confusion matrix:")
    print(train_cm)
    print("Validation confusion matrix:")
    print(val_cm)

    if test_ds is not None:
        test_metrics, test_cm = _evaluate_dataset(
            model, test_ds, num_classes
        )
        evaluations["test"] = (test_metrics, test_cm)
        test_bal_acc, test_bal_prec, test_bal_f1 = test_metrics
        print("Test report:")
        print(f"  Balanced accuracy: {test_bal_acc:.4f}")
        print(f"  Balanced precision: {test_bal_prec:.4f}")
        print(f"  Balanced f1-score: {test_bal_f1:.4f}")
        print("Test confusion matrix:")
        print(test_cm)

    _save_evaluation_csv(
        args.out_dir,
        args.model,
        args.fold,
        evaluations,
        class_names,
    )
    _save_confusion_matrix_plots(
        args.out_dir,
        args.model,
        args.fold,
        evaluations,
        class_names,
    )

    print(f"Finished training fold {args.fold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image classification training with stratified K-fold cross-validation"
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_dir2", type=str, default=None)
    parser.add_argument(
        "--data_dir3",
        type=str,
        default=None,
        help="Optional third channel root; files are matched by class and filename.",
    )
    parser.add_argument("--images_extension", type=str, default=".tif")
    parser.add_argument("--out_dir", type=str, default="outputs")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--model", type=str, default="custom",
                        choices=["custom", "resnet50", "convnext", "vgg16", "vit"])
    parser.add_argument(
        "--vit_patch_size",
        type=int,
        default=16,
        help="ViT patch size; the selected ViT-B/16 checkpoint requires 16.",
    )
    parser.add_argument("--vit_projection_dim", type=int, default=768)
    parser.add_argument("--vit_num_heads", type=int, default=12)
    parser.add_argument("--vit_transformer_layers", type=int, default=12)
    parser.add_argument("--vit_mlp_dim", type=int, default=3072)
    parser.add_argument("--mode", type=str, default="split",
                        choices=["split", "kfold"])
    parser.add_argument("--class_weight", type=str, default="balanced",
                        choices=["balanced", "none"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Frozen-backbone epochs before fine-tuning pretrained models.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-6,
        help="Learning rate used after unfreezing the pretrained backbone tail.",
    )
    parser.add_argument(
        "--unfreeze_fraction",
        type=float,
        default=0.05,
        help="Fraction of pretrained backbone layers to unfreeze.",
    )

    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, required=True)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
