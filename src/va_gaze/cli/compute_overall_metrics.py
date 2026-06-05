import argparse
import pandas as pd

from va_gaze.eval.oof_reports import create_prediction_tables


def main():
    parser = argparse.ArgumentParser(
        description="Recompute out-of-fold overall metrics from a Preds/<run> directory."
    )
    parser.add_argument(
        "preds_dir",
        help="Directory containing predictions_fold1.csv and predictions_fold2.csv.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing full_dataset_fold1.csv and full_dataset_fold2.csv.",
    )
    args = parser.parse_args()

    create_prediction_tables(args.preds_dir, data_dir=args.data_dir)
    overall = pd.read_csv(f"{args.preds_dir}/overall_metrics.csv")
    print("Out-of-fold overall metrics written.")
    print(f"Run dir: {args.preds_dir}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
