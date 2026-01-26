# imports
import argparse
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.models.image_based.dataloader.dataset_api import (
    build_stratified_kfold_benchmark,
    build_stratified_benchmark,
)

#################################################################################

def build_model(img_size, num_channels, num_classes, lr):
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

    return _compute_balanced_metrics(y_true, y_pred, num_classes)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    benchmark = build_stratified_kfold_benchmark(data_dir=args.data_dir,
                                                 batch_size=args.batch_size,
                                                 k=args.num_folds,
                                                 img_size=(args.img_size, args.img_size),
                                                 num_channels=args.num_channels,
                                                 data_dir2=args.data_dir2,
                                                 seed=args.seed
                                                 )

    fold_data = benchmark["folds"][args.fold]
    train_ds = fold_data["train"]
    val_ds = fold_data["val"]

    num_classes = benchmark["num_classes"]
    class_names = benchmark["class_names"]

    print(f"Fold {args.fold}/{args.num_folds - 1}")
    print(f"Classes ({num_classes}): {class_names}")

    if args.data_dir2 and args.num_channels != 2:
        print("Warning: data_dir2 provided but num_channels is not 2.")

    if args.num_channels == 3:
        def resnet_preprocess(x, y):
            return preprocess_input(x * 255.0), y

        train_ds = train_ds.map(resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    model = build_model(img_size=(args.img_size, args.img_size),
                        num_channels=args.num_channels,
                        num_classes=num_classes,
                        lr=args.lr
                        )

    model.summary()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.out_dir,
                                                    f"resnet50_fold{args.fold}.keras"),
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

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=callbacks
                        )

    bal_acc, bal_prec, bal_f1 = _evaluate_dataset(model, val_ds, num_classes)
    print("Validation report:")
    print(f"  Balanced accuracy: {bal_acc:.4f}")
    print(f"  Balanced precision: {bal_prec:.4f}")
    print(f"  Balanced f1-score: {bal_f1:.4f}")

    test_benchmark = build_stratified_benchmark(data_dir=args.data_dir,
                                                batch_size=args.batch_size,
                                                img_size=(args.img_size, args.img_size),
                                                num_channels=args.num_channels,
                                                data_dir2=args.data_dir2,
                                                seed=args.seed
                                                )
    test_ds = test_benchmark["test"]
    test_bal_acc, test_bal_prec, test_bal_f1 = _evaluate_dataset(
        model, test_ds, num_classes
    )
    print("Test report:")
    print(f"  Balanced accuracy: {test_bal_acc:.4f}")
    print(f"  Balanced precision: {test_bal_prec:.4f}")
    print(f"  Balanced f1-score: {test_bal_f1:.4f}")

    print(f"Finished training fold {args.fold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ResNet50 training with stratified K-fold cross-validation"
    )

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_dir2", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_channels", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, required=True)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
