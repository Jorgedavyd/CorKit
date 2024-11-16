import argparse
import asyncio
from .dataset import update


def main():
    parser = argparse.ArgumentParser(description="Corkit CLI dataset update manager.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for dataset updates (default: 10)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(update(batch_size=args.batch_size))
        print("Datasets updated successfully.")
    except Exception as e:
        print(f"Error during dataset update: {e}")
