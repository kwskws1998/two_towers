from va_gaze.train.fold_runner import run_fold


def training_fold2(model, loss, timestamp, params, dataset, preds_dir, checkpoint, gaze_config=None):
    train_data = dataset[1][0]
    val_data = dataset[1][1]
    run_fold(
        fold_id=2,
        model_name=model,
        loss_name=loss,
        timestamp=timestamp,
        params=params,
        train_data=train_data,
        val_data=val_data,
        preds_dir=preds_dir,
        checkpoint=checkpoint,
        prediction_filename="predictions_fold1.csv",
        metrics_filename="fold1_metrics.csv",
        gaze_config=gaze_config,
    )
