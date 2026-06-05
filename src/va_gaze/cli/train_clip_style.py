import argparse
import json
import os
import socket
from datetime import datetime

import pandas as pd
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, set_seed

from va_gaze.cli.setup_et_models import resolve_or_download_et_model2
from va_gaze.data.clip_dataset import VADTextDataset
from va_gaze.data.clip_splits import build_no_iemocap_splits
from va_gaze.eval.metrics import compute_metrics
from va_gaze.models.clip_alignment import GazeAffectClipModel
from va_gaze.train.clip_trainer import ClipAlignmentTrainer, numpy_json_safe


MODEL_CHOICES = ["distilbert", "xlmroberta-base", "xlmroberta-large"]
LOSS_CHOICES = ["mse", "ccc", "mse+ccc", "robust", "robust+ccc"]
MODEL_TO_CHECKPOINT = {
    "distilbert": "distilbert-base-multilingual-cased",
    "xlmroberta-base": "xlm-roberta-base",
    "xlmroberta-large": "xlm-roberta-large",
}


def parse_features_used(raw_value):
    try:
        parsed = [int(x.strip()) for x in str(raw_value).split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("features-used must be comma-separated integers.") from exc
    if len(parsed) != 5:
        raise argparse.ArgumentTypeError("features-used must contain 5 values: nFix,FFD,GPT,TRT,fixProp.")
    if any(value not in (0, 1) for value in parsed):
        raise argparse.ArgumentTypeError("features-used values must be 0 or 1.")
    if sum(parsed) == 0:
        raise argparse.ArgumentTypeError("features-used must enable at least one ET feature.")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train CLIP-style soft contrastive gaze-affect alignment for VA prediction."
    )
    parser.add_argument("model", choices=MODEL_CHOICES)
    parser.add_argument("loss", choices=LOSS_CHOICES)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-dir", default="data_clip_noiemocap")
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--holdout-dataset", default="IEMOCAP")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--et2-checkpoint", default="./checkpoints/et_predictor2_seed123")
    parser.add_argument("--no-et2-auto-download", action="store_true")
    parser.add_argument("--et2-hf-repo", default="skboy/et_prediction_2")
    parser.add_argument("--et2-hf-filename", default="et_predictor2_seed123.safetensors")
    parser.add_argument("--features-used", type=parse_features_used, default=parse_features_used("1,1,1,1,1"))
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--gaze-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--shuffle-gaze", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxlen", type=int, default=200)
    parser.add_argument("--output-root", default="PredsClip")
    parser.add_argument("--save-strategy", choices=["epoch", "no"], default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--save-final-model", dest="save_final_model", action="store_true")
    parser.add_argument("--no-save-final-model", dest="save_final_model", action="store_false")
    parser.set_defaults(save_final_model=True)
    return parser


def validate_args(parser, args):
    positive_ints = {
        "projection_dim": args.projection_dim,
        "gaze_hidden_dim": args.gaze_hidden_dim,
        "batch_size": args.batch_size,
        "train_epochs": args.train_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "maxlen": args.maxlen,
        "save_total_limit": args.save_total_limit,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            parser.error(f"{name} must be > 0.")
    if args.max_steps < -1 or args.max_steps == 0:
        parser.error("max_steps must be -1 or > 0.")
    if args.tau <= 0 or args.sigma <= 0:
        parser.error("tau and sigma must be > 0.")
    if args.lambda_align < 0:
        parser.error("lambda-align must be >= 0.")
    if args.dropout < 0 or args.dropout >= 1:
        parser.error("dropout must be in [0, 1).")


def create_run_dir(output_root):
    timestamp = datetime.now().strftime("%b-%d_%H-%M-%S")
    host_name = os.environ.get("COMPUTERNAME") or os.environ.get("HOST") or socket.gethostname()
    run_dir = os.path.join(output_root, f"{timestamp}_{host_name}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def build_training_args(args, run_dir):
    output_dir = os.path.join(run_dir, "checkpoints")
    logging_dir = os.path.join(run_dir, "logs")
    load_best_model_at_end = args.save_strategy != "no"
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim=args.optim,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        group_by_length=True,
        evaluation_strategy="epoch",
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        warmup_ratio=args.warmup_ratio,
        report_to=[],
    )


def write_predictions(path, source_df, predictions):
    df = source_df.copy().reset_index(drop=True)
    df = df.rename(columns={"valence": "valence_true", "arousal": "arousal_true"})
    df["valence_pred"] = predictions[:, 0]
    df["arousal_pred"] = predictions[:, 1]
    columns = [
        "index",
        "text",
        "dataset_of_origin",
        "valence_true",
        "arousal_true",
        "valence_pred",
        "arousal_pred",
    ]
    df[columns].to_csv(path, index=False)


def predict_and_save(trainer, dataset, run_dir, name):
    output = trainer.predict(dataset, metric_key_prefix=name)
    predictions_path = os.path.join(run_dir, f"predictions_{name}.csv")
    metrics_path = os.path.join(run_dir, f"metrics_{name}.json")
    write_predictions(predictions_path, dataset.df, output.predictions)
    with open(metrics_path, "w") as output_file:
        json.dump(numpy_json_safe(output.metrics), output_file, indent=2)
    return output.metrics


def save_run_parameters(run_dir, args, checkpoint, et2_checkpoint, split_paths):
    params = vars(args).copy()
    params["checkpoint"] = checkpoint
    params["resolved_et2_checkpoint"] = et2_checkpoint
    params["split_paths"] = {name: str(path) for name, path in split_paths.items()}
    with open(os.path.join(run_dir, "training_parameters.json"), "w") as output_file:
        json.dump(params, output_file, indent=2)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    set_seed(args.seed)

    checkpoint = MODEL_TO_CHECKPOINT[args.model]
    split_paths = build_no_iemocap_splits(
        data_dir=args.data_dir,
        output_dir=args.split_dir,
        holdout_dataset=args.holdout_dataset,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        force=args.force_splits,
    )
    et2_checkpoint = resolve_or_download_et_model2(
        args.et2_checkpoint,
        auto_download=not args.no_et2_auto_download,
        hf_repo_id=args.et2_hf_repo,
        hf_filename=args.et2_hf_filename,
    )

    run_dir = create_run_dir(args.output_root)
    save_run_parameters(run_dir, args, checkpoint, et2_checkpoint, split_paths)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_dataset = VADTextDataset(split_paths["train"], checkpoint, args.maxlen, tokenizer=tokenizer)
    val_dataset = VADTextDataset(split_paths["val"], checkpoint, args.maxlen, tokenizer=tokenizer)
    test_dataset = VADTextDataset(split_paths["test"], checkpoint, args.maxlen, tokenizer=tokenizer)
    holdout_dataset = VADTextDataset(split_paths["holdout"], checkpoint, args.maxlen, tokenizer=tokenizer)

    model = GazeAffectClipModel(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        et2_checkpoint_path=et2_checkpoint,
        features_used=args.features_used,
        projection_dim=args.projection_dim,
        gaze_hidden_dim=args.gaze_hidden_dim,
        dropout=args.dropout,
        shuffle_gaze=args.shuffle_gaze,
    )
    training_args = build_training_args(args, run_dir)
    trainer = ClipAlignmentTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        loss_name=args.loss,
        lambda_align=args.lambda_align,
        tau=args.tau,
        sigma=args.sigma,
    )

    trainer.train()
    if args.save_final_model:
        trainer.save_model(os.path.join(run_dir, "final_model"))

    metrics = {
        "val": predict_and_save(trainer, val_dataset, run_dir, "noiemocap_val"),
        "test": predict_and_save(trainer, test_dataset, run_dir, "noiemocap_test"),
        "iemocap_zeroshot": predict_and_save(trainer, holdout_dataset, run_dir, "iemocap_zeroshot"),
    }
    with open(os.path.join(run_dir, "metrics_summary.json"), "w") as output_file:
        json.dump({key: numpy_json_safe(value) for key, value in metrics.items()}, output_file, indent=2)

    print(f"Saved CLIP-style run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
