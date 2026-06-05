import argparse
import json
import os
import socket
from datetime import datetime

from transformers import DataCollatorWithPadding, TrainingArguments, set_seed

from va_gaze.data.clip_dataset import VADTextDataset
from va_gaze.data.clip_splits import build_no_iemocap_splits
from va_gaze.eval.metrics import compute_metrics
from va_gaze.models.regression import (
    DistilBertForSequenceClassificationSig,
    XLMRobertaForSequenceClassificationSig,
)
from va_gaze.train.clip_trainer import numpy_json_safe
from va_gaze.train.custom_trainer import (
    CustomTrainerCCC,
    CustomTrainerMSE,
    CustomTrainerMSE_CCC,
    CustomTrainerRobust,
    CustomTrainerRobustCCC,
)


MODEL_CHOICES = ["distilbert", "xlmroberta-base", "xlmroberta-large"]
LOSS_CHOICES = ["mse", "ccc", "mse+ccc", "robust", "robust+ccc"]
MODEL_TO_CHECKPOINT = {
    "distilbert": "distilbert-base-multilingual-cased",
    "xlmroberta-base": "xlm-roberta-base",
    "xlmroberta-large": "xlm-roberta-large",
}
LOSS_TO_TRAINER = {
    "mse": CustomTrainerMSE,
    "ccc": CustomTrainerCCC,
    "mse+ccc": CustomTrainerMSE_CCC,
    "robust": CustomTrainerRobust,
    "robust+ccc": CustomTrainerRobustCCC,
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train one text-only VAD model with no-IEMOCAP train/val/test and IEMOCAP zero-shot eval."
    )
    parser.add_argument("model", choices=MODEL_CHOICES)
    parser.add_argument("loss", choices=LOSS_CHOICES)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-dir", default="data_vad_noiemocap")
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--holdout-dataset", default="IEMOCAP")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
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
    parser.add_argument("--output-root", default="PredsVADSingle")
    parser.add_argument("--model-root", default="model_vad_single")
    parser.add_argument("--save-strategy", choices=["epoch", "no"], default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--save-final-model", dest="save_final_model", action="store_true")
    parser.add_argument("--no-save-final-model", dest="save_final_model", action="store_false")
    parser.set_defaults(save_final_model=True)
    return parser


def validate_args(parser, args):
    for name in [
        "batch_size",
        "train_epochs",
        "gradient_accumulation_steps",
        "maxlen",
        "save_total_limit",
    ]:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be > 0.")
    if args.max_steps < -1 or args.max_steps == 0:
        parser.error("max_steps must be -1 or > 0.")
    if args.val_size <= 0 or args.test_size <= 0 or args.val_size + args.test_size >= 1:
        parser.error("val-size and test-size must be positive and sum to less than 1.")


def create_run_dir(output_root):
    timestamp = datetime.now().strftime("%b-%d_%H-%M-%S")
    host_name = os.environ.get("COMPUTERNAME") or os.environ.get("HOST") or socket.gethostname()
    run_dir = os.path.join(output_root, f"{timestamp}_{host_name}")
    os.makedirs(run_dir, exist_ok=False)
    return timestamp, run_dir


def build_model(model_name, checkpoint):
    if model_name == "distilbert":
        return DistilBertForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    if model_name in ("xlmroberta-base", "xlmroberta-large"):
        return XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    raise ValueError(f"Unknown model name: {model_name}")


def build_training_args(args, run_dir):
    load_best_model_at_end = args.save_strategy != "no"
    return TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        logging_dir=os.path.join(run_dir, "logs"),
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
    write_predictions(os.path.join(run_dir, f"predictions_{name}.csv"), dataset.df, output.predictions)
    with open(os.path.join(run_dir, f"metrics_{name}.json"), "w") as output_file:
        json.dump(numpy_json_safe(output.metrics), output_file, indent=2)
    return output.metrics


def save_run_parameters(run_dir, args, checkpoint, split_paths, final_model_dir):
    params = vars(args).copy()
    params["checkpoint"] = checkpoint
    params["split_paths"] = {name: str(path) for name, path in split_paths.items()}
    params["final_model_dir"] = final_model_dir
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
    timestamp, run_dir = create_run_dir(args.output_root)
    final_model_dir = os.path.join(args.model_root, timestamp, "final_model")
    save_run_parameters(run_dir, args, checkpoint, split_paths, final_model_dir)

    train_dataset = VADTextDataset(split_paths["train"], checkpoint, args.maxlen)
    val_dataset = VADTextDataset(
        split_paths["val"],
        checkpoint,
        args.maxlen,
        tokenizer=train_dataset.tokenizer,
    )
    test_dataset = VADTextDataset(
        split_paths["test"],
        checkpoint,
        args.maxlen,
        tokenizer=train_dataset.tokenizer,
    )
    holdout_dataset = VADTextDataset(
        split_paths["holdout"],
        checkpoint,
        args.maxlen,
        tokenizer=train_dataset.tokenizer,
    )

    model = build_model(args.model, checkpoint)
    trainer_cls = LOSS_TO_TRAINER[args.loss]
    trainer = trainer_cls(
        model,
        build_training_args(args, run_dir),
        data_collator=DataCollatorWithPadding(train_dataset.tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=train_dataset.tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    if args.save_final_model:
        trainer.save_model(final_model_dir)

    metrics = {
        "val": predict_and_save(trainer, val_dataset, run_dir, "noiemocap_val"),
        "test": predict_and_save(trainer, test_dataset, run_dir, "noiemocap_test"),
        "iemocap_zeroshot": predict_and_save(trainer, holdout_dataset, run_dir, "iemocap_zeroshot"),
    }
    with open(os.path.join(run_dir, "metrics_summary.json"), "w") as output_file:
        json.dump({key: numpy_json_safe(value) for key, value in metrics.items()}, output_file, indent=2)

    print(f"Saved single VAD run outputs to: {run_dir}")
    if args.save_final_model:
        print(f"Saved single VAD model to: {final_model_dir}")


if __name__ == "__main__":
    main()
