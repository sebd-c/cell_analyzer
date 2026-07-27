# imports
import argparse
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, LayerNormalization,
                                     GlobalAveragePooling1D, MultiHeadAttention,
                                     Embedding, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

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
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_vit_model(img_size, num_channels, num_classes, lr, patch_size=16,
                    projection_dim=64, num_heads=4, transformer_layers=8,
                    mlp_dim=128):
    """Build a Vision Transformer classifier trained from scratch.

    Unlike the ImageNet application backbones, this model accepts any positive
    number of channels and consumes the dataset's existing [0, 1] float range.
    """
    height, width = img_size
    if patch_size <= 0:
        raise ValueError("ViT patch_size must be a positive integer.")
    if height % patch_size or width % patch_size:
        raise ValueError(
            "ViT patch_size must evenly divide both image dimensions; got "
            f"image size {img_size} and patch_size {patch_size}."
        )
    if num_heads <= 0:
        raise ValueError("ViT num_heads must be a positive integer.")
    if projection_dim % num_heads:
        raise ValueError("ViT projection_dim must be divisible by num_heads.")
    if transformer_layers <= 0:
        raise ValueError("ViT transformer_layers must be a positive integer.")
    if mlp_dim <= 0:
        raise ValueError("ViT mlp_dim must be a positive integer.")

    num_patches = (height // patch_size) * (width // patch_size)
    inputs = tf.keras.Input(shape=(height, width, num_channels))
    patches = Conv2D(projection_dim, kernel_size=patch_size,
                     strides=patch_size, padding="valid")(inputs)
    tokens = tf.keras.layers.Reshape((num_patches, projection_dim))(patches)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = Embedding(input_dim=num_patches,
                                   output_dim=projection_dim)(positions)
    x = Add()([tokens, position_embedding])

    for index in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6,
                                name=f"transformer_{index}_norm_1")(x)
        attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=0.1,
            name=f"transformer_{index}_attention",
        )(x1, x1)
        x2 = Add(name=f"transformer_{index}_attention_residual")([x, attention])

        x3 = LayerNormalization(epsilon=1e-6,
                                name=f"transformer_{index}_norm_2")(x2)
        x3 = Dense(mlp_dim, activation=tf.nn.gelu,
                   name=f"transformer_{index}_mlp_1")(x3)
        x3 = Dropout(0.1, name=f"transformer_{index}_mlp_dropout")(x3)
        x3 = Dense(projection_dim, name=f"transformer_{index}_mlp_2")(x3)
        x = Add(name=f"transformer_{index}_mlp_residual")([x2, x3])

    x = LayerNormalization(epsilon=1e-6, name="encoder_norm")(x)
    x = GlobalAveragePooling1D(name="token_pooling")(x)
    x = Dropout(0.2, name="classifier_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="classifier")(x)
    model = Model(inputs=inputs, outputs=outputs, name="vision_transformer")
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_custom_cnn(img_size, num_channels, num_classes, lr):
    inputs = tf.keras.Input(shape=(*img_size, num_channels))
    x = Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, 3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
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

    effective_channels = args.num_channels
    imagenet_models = {"resnet50", "convnext"}
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
        print("Warning: data_dir2 provided but num_channels is not 2.")

    if args.model in imagenet_models:
        def imagenet_preprocess(x, y):
            # The dataset API emits float images in [0, 1]. Both ResNet50 and
            # ConvNeXt ImageNet backbones expect values expressed in [0, 255].
            if args.model == "resnet50":
                x = preprocess_input(x * 255.0)
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
    else:
        model = build_custom_cnn(img_size=(args.img_size, args.img_size),
                                 num_channels=effective_channels,
                                 num_classes=num_classes,
                                 lr=args.lr
                                 )

    model.summary()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.out_dir,
                                                    f"{args.model}_fold{args.fold}.keras"),
                                                    monitor="val_loss",
                                                    save_best_only=True
                                                    ),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                  patience=5,
                                                  restore_best_weights=True
                                                  )
                 ,
                 _EpochLossPrinter()
                 ]

    class_weight = None
    if args.class_weight == "balanced":
        counts, weights = _compute_class_weights(train_samples, num_classes)
        print(f"Train class counts: {counts.tolist()}")
        print(f"Using balanced class weights: {weights}")
        class_weight = weights

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        class_weight=class_weight
                        )

    train_metrics, train_cm = _evaluate_dataset(model, train_ds, num_classes)
    val_metrics, val_cm = _evaluate_dataset(model, val_ds, num_classes)
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
        test_bal_acc, test_bal_prec, test_bal_f1 = test_metrics
        print("Test report:")
        print(f"  Balanced accuracy: {test_bal_acc:.4f}")
        print(f"  Balanced precision: {test_bal_prec:.4f}")
        print(f"  Balanced f1-score: {test_bal_f1:.4f}")
        print("Test confusion matrix:")
        print(test_cm)

    print(f"Finished training fold {args.fold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image classification training with stratified K-fold cross-validation"
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_dir2", type=str, default=None)
    parser.add_argument("--images_extension", type=str, default=".tif")
    parser.add_argument("--out_dir", type=str, default="outputs")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--model", type=str, default="custom",
                        choices=["custom", "resnet50", "convnext", "vit"])
    parser.add_argument("--vit_patch_size", type=int, default=16)
    parser.add_argument("--vit_projection_dim", type=int, default=64)
    parser.add_argument("--vit_num_heads", type=int, default=4)
    parser.add_argument("--vit_transformer_layers", type=int, default=8)
    parser.add_argument("--vit_mlp_dim", type=int, default=128)
    parser.add_argument("--mode", type=str, default="split",
                        choices=["split", "kfold"])
    parser.add_argument("--class_weight", type=str, default="balanced",
                        choices=["balanced", "none"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, required=True)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
