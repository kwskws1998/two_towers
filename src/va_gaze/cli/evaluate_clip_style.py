import argparse
import json
import os
import socket
from datetime import datetime

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, set_seed

from va_gaze.cli.setup_et_models import resolve_or_download_et_model2
from va_gaze.cli.train_clip_style import (
    LOSS_CHOICES,
    MODEL_CHOICES,
    MODEL_TO_CHECKPOINT,
    parse_features_used,
    predict_and_save,
)
from va_gaze.data.clip_dataset import VADTextDataset
from va_gaze.data.clip_splits import build_no_iemocap_splits
from va_gaze.eval.metrics import compute_metrics
from va_gaze.models.clip_alignment import GazeAffectClipModel
from va_gaze.train.clip_trainer import ClipAlignmentTrainer, numpy_json_safe


SPLIT_TO_KEY = {
    "noiemocap_val": "val",
    "noiemocap_test": "test",
    "iemocap_zeroshot": "holdout",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CLIP-style gaze-affect model without training."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, default=None)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-dir", default="data_clip_noiemocap")
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--holdout-dataset", default="IEMOCAP")
    parser.add_argument("--split", choices=["noiemocap_val", "noiemocap_test", "iemocap_zeroshot", "all"], default="iemocap_zeroshot")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--et2-checkpoint", default="./checkpoints/et_predictor2_seed123")
    parser.add_argument("--no-et2-auto-download", action="store_true")
    parser.add_argument("--et2-hf-repo", default="skboy/et_prediction_2")
    parser.add_argument("--et2-hf-filename", default="et_predictor2_seed123.safetensors")
    parser.add_argument("--features-used", type=parse_features_used, default=None)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--gaze-hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--lambda-align", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxlen", type=int, default=None)
    parser.add_argument("--output-root", default="PredsClipEval")
    return parser


def create_run_dir(output_root):
    timestamp = datetime.now().strftime("%b-%d_%H-%M-%S")
    host_name = os.environ.get("COMPUTERNAME") or os.environ.get("HOST") or socket.gethostname()
    run_dir = os.path.join(output_root, f"{timestamp}_{host_name}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def load_params(checkpoint_dir):
    path = os.path.join(checkpoint_dir, "training_parameters.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as input_file:
        return json.load(input_file)


def choose_arg(args, params, name, default):
    value = getattr(args, name)
    if value is not None:
        return value
    return params.get(name, default)


def resolve_state_path(checkpoint_dir):
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}")


def load_model_state(model, checkpoint_dir):
    state_path = resolve_state_path(checkpoint_dir)
    if state_path.endswith(".safetensors"):
        state = load_safetensors(state_path, device="cpu")
    else:
        state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if not key.startswith("fp_model.")]
    unexpected = [key for key in unexpected if not key.startswith("fp_model.")]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint did not match GazeAffectClipModel. "
            f"Missing keys: {missing[:20]} Unexpected keys: {unexpected[:20]}"
        )
    print(f"[evaluate_clip_style] loaded model state: {state_path}")


def build_eval_args(args, run_dir):
    return TrainingArguments(
        output_dir=os.path.join(run_dir, "trainer_tmp"),
        per_device_eval_batch_size=args.batch_size,
        dataloader_drop_last=False,
        seed=args.seed,
        report_to=[],
    )


def save_eval_parameters(run_dir, args, resolved):
    payload = vars(args).copy()
    payload.update(resolved)
    with open(os.path.join(run_dir, "eval_parameters.json"), "w") as output_file:
        json.dump(numpy_json_safe(payload), output_file, indent=2)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be > 0.")
    set_seed(args.seed)

    params = load_params(args.checkpoint_dir)
    model_choice = args.model or params.get("model")
    if model_choice not in MODEL_TO_CHECKPOINT:
        parser.error("--model is required when training_parameters.json is missing or incomplete.")

    loss_name = args.loss or params.get("loss", "mse")
    checkpoint = MODEL_TO_CHECKPOINT[model_choice]
    features_used = args.features_used or params.get("features_used", [1, 1, 1, 1, 1])
    projection_dim = choose_arg(args, params, "projection_dim", 256)
    gaze_hidden_dim = choose_arg(args, params, "gaze_hidden_dim", 256)
    dropout = choose_arg(args, params, "dropout", 0.1)
    tau = choose_arg(args, params, "tau", 0.07)
    sigma = choose_arg(args, params, "sigma", 0.05)
    lambda_align = choose_arg(args, params, "lambda_align", 0.1)
    maxlen = choose_arg(args, params, "maxlen", 200)

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
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    model = GazeAffectClipModel(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        et2_checkpoint_path=et2_checkpoint,
        features_used=features_used,
        projection_dim=projection_dim,
        gaze_hidden_dim=gaze_hidden_dim,
        dropout=dropout,
        shuffle_gaze=False,
        init_vad_checkpoint=None,
    )
    load_model_state(model, args.checkpoint_dir)

    trainer = ClipAlignmentTrainer(
        model=model,
        args=build_eval_args(args, run_dir),
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        loss_name=loss_name,
        lambda_align=lambda_align,
        tau=tau,
        sigma=sigma,
    )

    selected = list(SPLIT_TO_KEY.keys()) if args.split == "all" else [args.split]
    metrics = {}
    for split_name in selected:
        dataset = VADTextDataset(split_paths[SPLIT_TO_KEY[split_name]], checkpoint, maxlen, tokenizer=tokenizer)
        metrics[split_name] = predict_and_save(trainer, dataset, run_dir, split_name)

    save_eval_parameters(
        run_dir,
        args,
        {
            "checkpoint": checkpoint,
            "resolved_et2_checkpoint": et2_checkpoint,
            "features_used": features_used,
            "projection_dim": projection_dim,
            "gaze_hidden_dim": gaze_hidden_dim,
            "dropout": dropout,
            "tau": tau,
            "sigma": sigma,
            "lambda_align": lambda_align,
            "maxlen": maxlen,
            "split_paths": {name: str(path) for name, path in split_paths.items()},
        },
    )
    with open(os.path.join(run_dir, "metrics_summary.json"), "w") as output_file:
        json.dump({key: numpy_json_safe(value) for key, value in metrics.items()}, output_file, indent=2)

    print(f"Saved CLIP-style eval outputs to: {run_dir}")


if __name__ == "__main__":
    main()
