# imports
import argparse
import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.models.image_based.dataloader.dataset_api import build_stratified_kfold_benchmark

#################################################################################

def build_model(img_size, num_channels, num_classes, lr):
    base_model = ResNet50(weights="imagenet",
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


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    benchmark = build_stratified_kfold_benchmark(data_dir=args.data_dir,
                                                 batch_size=args.batch_size,
                                                 k=args.num_folds,
                                                 img_size=(args.img_size, args.img_size),
                                                 num_channels=args.num_channels,
                                                 seed=args.seed
                                                 )

    fold_data = benchmark["folds"][args.fold]
    train_ds = fold_data["train"]
    val_ds = fold_data["val"]

    num_classes = benchmark["num_classes"]
    class_names = benchmark["class_names"]

    print(f"Fold {args.fold}/{args.num_folds - 1}")
    print(f"Classes ({num_classes}): {class_names}")

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
                 ]

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=callbacks
                        )

    print(f"Finished training fold {args.fold}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ResNet50 training with stratified K-fold cross-validation"
    )

    parser.add_argument("--data_dir", type=str, required=True)
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
