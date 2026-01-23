import argparse
from dataset_api import build_benchmark_datasets

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build benchmark-ready TF datasets from TIFF folders"
    )

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory with class subfolders")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--num_channels", type=int, default=1)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def main():
    args = parse_args()

    datasets = build_benchmark_datasets(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_channels=args.num_channels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print("✔ Dataset built successfully")
    print("Classes:", datasets["class_names"])
    print("Num classes:", datasets["num_classes"])

    # Sanity check
    for x, y in datasets["train"].take(1):
        print("Train batch shape:", x.shape)
        print("Train labels:", y.numpy())


if __name__ == "__main__":
    main()
