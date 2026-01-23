import argparse
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from dataset_api import get_kfold_datasets


def build_model(num_classes, img_size, lr):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, 3)
    )

    base_model.trainable = False  # transfer learning first

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main(args):
    print(f"Running fold {args.fold}/{args.num_folds}")

    train_ds, val_ds, class_names = get_kfold_datasets(
        root_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        fold_idx=args.fold,
        seed=args.seed
    )

    model = build_model(
        num_classes=len(class_names),
        img_size=(args.img_size, args.img_size),
        lr=args.lr
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{args.out_dir}/resnet50_fold{args.fold}.keras",
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ResNet50 training with K-fold CV")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
