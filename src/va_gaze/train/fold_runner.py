import pandas as pd
from transformers import DataCollatorWithPadding, TrainingArguments

from va_gaze.train.custom_trainer import (
    CustomTrainerCCC,
    CustomTrainerMSE,
    CustomTrainerMSE_CCC,
    CustomTrainerRobust,
    CustomTrainerRobustCCC,
)
from va_gaze.eval.metrics import compute_metrics
from va_gaze.models.regression import (
    DistilBertForSequenceClassificationSig,
    GazeAddForSequenceRegression,
    GazeConcatForSequenceRegression,
    XLMRobertaForSequenceClassificationSig,
)


LOSS_TO_TRAINER = {
    "mse": CustomTrainerMSE,
    "ccc": CustomTrainerCCC,
    "robust": CustomTrainerRobust,
    "mse+ccc": CustomTrainerMSE_CCC,
    "robust+ccc": CustomTrainerRobustCCC,
}


def _select_batch_size(model_name, params):
    if model_name == "distilbert":
        return params["batch_size_distil"]
    if model_name == "xlmroberta-base":
        return params["batch_size_xlmrB"]
    if model_name == "xlmroberta-large":
        return params["batch_size_xlmrL"]
    raise ValueError(f"Unknown model name: {model_name}")


def _build_model(model_name, checkpoint, tokenizer, gaze_config):
    use_gaze_concat = bool(gaze_config.get("use_gaze_concat", False))
    use_gaze_add = bool(gaze_config.get("use_gaze_add", False))
    if use_gaze_concat:
        return GazeConcatForSequenceRegression(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=gaze_config.get("et2_checkpoint_path"),
            features_used=gaze_config.get("features_used", [1, 1, 1, 1, 1]),
            fp_dropout=tuple(gaze_config.get("fp_dropout", [0.0, 0.3])),
        )
    if use_gaze_add:
        return GazeAddForSequenceRegression(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=gaze_config.get("et2_checkpoint_path"),
            features_used=gaze_config.get("features_used", [1, 1, 1, 1, 1]),
            fp_dropout=tuple(gaze_config.get("fp_dropout", [0.0, 0.3])),
            gaze_add_scale=gaze_config.get("gaze_add_scale", 0.05),
            train_gaze_add_scale=gaze_config.get("train_gaze_add_scale", False),
        )

    if model_name == "distilbert":
        return DistilBertForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    if model_name in ("xlmroberta-base", "xlmroberta-large"):
        return XLMRobertaForSequenceClassificationSig.from_pretrained(checkpoint, num_labels=2)
    raise ValueError(f"Unknown model name: {model_name}")


def _build_training_args(output_dir, logging_dir, batch_size, params):
    save_strategy = params.get("save_strategy", "epoch")
    load_best_model_at_end = params.get("load_best_model_at_end", True)
    if save_strategy == "no":
        load_best_model_at_end = False

    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=200,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=params["train_epochs"],
        max_steps=params.get("max_steps", -1),
        learning_rate=params["lr"],
        weight_decay=params["weight_decay"],
        optim=params.get("optim", "adamw_torch"),
        gradient_accumulation_steps=params.get("gradient_accumulation_steps", 1),
        seed=params.get("seed", 42),
        group_by_length=True,
        evaluation_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=params.get("save_total_limit", 1),
        load_best_model_at_end=load_best_model_at_end,
        warmup_ratio=params["warmup_ratio"],
    )


def _build_trainer(loss_name, model, training_args, train_data, val_data):
    trainer_cls = LOSS_TO_TRAINER.get(loss_name)
    if trainer_cls is None:
        raise ValueError(f"Unknown loss name: {loss_name}")
    return trainer_cls(
        model,
        training_args,
        data_collator=DataCollatorWithPadding(train_data.tokenizer),
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=train_data.tokenizer,
        compute_metrics=compute_metrics,
    )


def run_fold(
    fold_id,
    model_name,
    loss_name,
    timestamp,
    params,
    train_data,
    val_data,
    preds_dir,
    checkpoint,
    prediction_filename,
    metrics_filename,
    gaze_config=None,
):
    gaze_config = gaze_config or {}
    output_dir = f"Output Directory/{timestamp}/fold{fold_id}"
    model_dir = f"model/{timestamp}/fold{fold_id}"
    logging_dir = f"logs/logs{fold_id}"
    batch_size = _select_batch_size(model_name, params)

    model = _build_model(model_name, checkpoint, train_data.tokenizer, gaze_config)
    training_args = _build_training_args(output_dir, logging_dir, batch_size, params)
    trainer = _build_trainer(loss_name, model, training_args, train_data, val_data)

    print(f"Starting fold {fold_id}")
    trainer.train()
    predictions = trainer.predict(val_data)

    pd.DataFrame(predictions.predictions).to_csv(f"{preds_dir}/{prediction_filename}")
    with open(f"{preds_dir}/{metrics_filename}", "w") as output_file:
        for key, value in predictions.metrics.items():
            output_file.write(f"{key},{value}\n")

    if params.get("save_final_model", True):
        trainer.save_model(model_dir)
