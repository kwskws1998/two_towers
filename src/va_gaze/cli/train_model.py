import argparse
import json
import os
import socket
from datetime import datetime
from signal import signal

from va_gaze.data.dataset import MyDataset
from va_gaze.eval.oof_reports import create_prediction_tables, handle_signal, set_preds_dir
from va_gaze.train.fold1 import training_fold1
from va_gaze.train.fold2 import training_fold2


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL_CHOICES = ["distilbert", "xlmroberta-base", "xlmroberta-large"]
LOSS_CHOICES = ["mse", "ccc", "robust", "mse+ccc", "robust+ccc"]
MODEL_TO_CHECKPOINT = {
    "distilbert": "distilbert-base-multilingual-cased",
    "xlmroberta-base": "xlm-roberta-base",
    "xlmroberta-large": "xlm-roberta-large",
}


def _parse_features_used(raw_value):
    try:
        parsed = [int(x.strip()) for x in str(raw_value).split(",")]
    except ValueError as exc:
        raise ValueError("features_used must be a comma-separated list of integers.") from exc
    if len(parsed) != 5:
        raise ValueError("features_used must contain exactly 5 values (nFix,FFD,GPT,TRT,fixProp).")
    if any(value not in (0, 1) for value in parsed):
        raise ValueError("features_used values must be 0 or 1.")
    if sum(parsed) == 0:
        raise ValueError("At least one gaze feature must be enabled in features_used.")
    return parsed


def _parse_fp_dropout(raw_value):
    try:
        parsed = [float(x.strip()) for x in str(raw_value).split(",")]
    except ValueError as exc:
        raise ValueError("fp_dropout must be a comma-separated list of floats.") from exc
    if len(parsed) != 2:
        raise ValueError("fp_dropout must contain exactly 2 values.")
    return parsed


def _validate_positive_int(name, value):
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODEL_CHOICES)
    parser.add_argument("loss", choices=LOSS_CHOICES)
    parser.add_argument("--use-gaze-concat", action="store_true")
    parser.add_argument("--use-gaze-add", action="store_true")
    parser.add_argument("--et2-checkpoint", default=None)
    parser.add_argument("--features-used", default="1,1,1,1,1")
    parser.add_argument("--fp-dropout", default="0.0,0.3")
    parser.add_argument("--gaze-add-scale", type=float, default=0.05)
    parser.add_argument("--train-gaze-add-scale", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-size-distil", type=int, default=None)
    parser.add_argument("--batch-size-xlmrb", dest="batch_size_xlmrB", type=int, default=None)
    parser.add_argument("--batch-size-xlmrl", dest="batch_size_xlmrL", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--save-strategy", choices=["epoch", "no"], default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--save-final-model", dest="save_final_model", action="store_true")
    parser.add_argument("--no-save-final-model", dest="save_final_model", action="store_false")
    parser.add_argument(
        "--load-best-model-at-end",
        dest="load_best_model_at_end",
        action="store_true",
    )
    parser.add_argument(
        "--no-load-best-model-at-end",
        dest="load_best_model_at_end",
        action="store_false",
    )
    parser.set_defaults(load_best_model_at_end=True)
    parser.set_defaults(save_final_model=True)
    return parser


def _validate_args(parser, args):
    try:
        features_used = _parse_features_used(args.features_used)
        fp_dropout = _parse_fp_dropout(args.fp_dropout)
        _validate_positive_int("train_epochs", args.train_epochs)
        _validate_positive_int("gradient_accumulation_steps", args.gradient_accumulation_steps)
        _validate_positive_int("maxlen", args.maxlen)
        _validate_positive_int("save_total_limit", args.save_total_limit)
        if args.max_steps < -1 or args.max_steps == 0:
            raise ValueError("max_steps must be -1 or > 0.")
        if args.batch_size is not None:
            _validate_positive_int("batch_size", args.batch_size)
        if args.batch_size_distil is not None:
            _validate_positive_int("batch_size_distil", args.batch_size_distil)
        if args.batch_size_xlmrB is not None:
            _validate_positive_int("batch_size_xlmrB", args.batch_size_xlmrB)
        if args.batch_size_xlmrL is not None:
            _validate_positive_int("batch_size_xlmrL", args.batch_size_xlmrL)
    except ValueError as exc:
        parser.error(str(exc))

    if args.gaze_add_scale < 0:
        parser.error("gaze_add_scale must be >= 0.")

    if args.use_gaze_concat and args.use_gaze_add:
        parser.error("--use-gaze-concat and --use-gaze-add are mutually exclusive.")

    if args.use_gaze_concat and args.maxlen > 255:
        parser.error(
            "When --use-gaze-concat is enabled, maxlen must be <= 255 to avoid positional limit overflow."
        )

    if args.save_strategy == "no" and args.load_best_model_at_end:
        args.load_best_model_at_end = False
        print("[train_model] save_strategy=no, so load_best_model_at_end was set to False.")

    return features_used, fp_dropout


def _resolve_batch_sizes(args):
    base_batch_size = args.batch_size if args.batch_size is not None else 16
    batch_size_distil = args.batch_size_distil if args.batch_size_distil is not None else base_batch_size
    batch_size_xlmrB = args.batch_size_xlmrB if args.batch_size_xlmrB is not None else base_batch_size
    batch_size_xlmrL = args.batch_size_xlmrL if args.batch_size_xlmrL is not None else base_batch_size
    return batch_size_distil, batch_size_xlmrB, batch_size_xlmrL


def _create_run_dir():
    timestamp = datetime.now().strftime("%b-%d_%H-%M-%S")
    host_name = os.environ.get("COMPUTERNAME") or os.environ.get("HOST") or socket.gethostname()
    preds_dir = f"Preds/{timestamp}_{host_name}"
    os.makedirs(preds_dir)
    set_preds_dir(preds_dir)
    return timestamp, preds_dir


def _save_training_parameters(preds_dir, run_parameters):
    with open(f"{preds_dir}/training_parameters.json", "w") as output_file:
        json.dump(run_parameters, output_file)


def _load_dataset(checkpoint, maxlen, data_dir):
    filename_1 = os.path.join(data_dir, "full_dataset_fold1.csv")
    filename_2 = os.path.join(data_dir, "full_dataset_fold2.csv")
    split_1 = MyDataset(filename=filename_1, checkpoint=checkpoint, maxlen=maxlen)
    split_2 = MyDataset(filename=filename_2, checkpoint=checkpoint, maxlen=maxlen)
    return [[split_1, split_2], [split_2, split_1]]


def main():
    signal(2, handle_signal)
    parser = _build_parser()
    args = parser.parse_args()

    features_used, fp_dropout = _validate_args(parser, args)
    batch_size_distil, batch_size_xlmrB, batch_size_xlmrL = _resolve_batch_sizes(args)
    checkpoint = MODEL_TO_CHECKPOINT[args.model]
    gaze_config = {
        "use_gaze_concat": args.use_gaze_concat,
        "use_gaze_add": args.use_gaze_add,
        "et2_checkpoint_path": args.et2_checkpoint,
        "features_used": features_used,
        "fp_dropout": fp_dropout,
        "gaze_add_scale": args.gaze_add_scale,
        "train_gaze_add_scale": args.train_gaze_add_scale,
    }

    timestamp, preds_dir = _create_run_dir()
    params = {
        "batch_size_distil": batch_size_distil,
        "batch_size_xlmrB": batch_size_xlmrB,
        "batch_size_xlmrL": batch_size_xlmrL,
        "lr": args.learning_rate,
        "train_epochs": args.train_epochs,
        "max_steps": args.max_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "optim": args.optim,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "maxlen": args.maxlen,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "save_final_model": args.save_final_model,
        "load_best_model_at_end": args.load_best_model_at_end,
        "data_dir": args.data_dir,
    }
    run_parameters = {
        "model": args.model,
        "loss_function": args.loss,
        "use_gaze_concat": gaze_config["use_gaze_concat"],
        "use_gaze_add": gaze_config["use_gaze_add"],
        "et2_checkpoint_path": gaze_config["et2_checkpoint_path"],
        "features_used": gaze_config["features_used"],
        "fp_dropout": gaze_config["fp_dropout"],
        "gaze_add_scale": gaze_config["gaze_add_scale"],
        "train_gaze_add_scale": gaze_config["train_gaze_add_scale"],
        "path": preds_dir,
        **params,
    }
    _save_training_parameters(preds_dir, run_parameters)

    dataset = _load_dataset(checkpoint, args.maxlen, args.data_dir)
    training_fold1(args.model, args.loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=gaze_config)
    print("\n\n\n------------ NOW ON FOLD 2 -------------- \n\n\n")
    training_fold2(args.model, args.loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=gaze_config)
    create_prediction_tables(preds_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
