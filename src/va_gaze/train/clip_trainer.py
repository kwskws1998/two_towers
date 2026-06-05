import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer

from va_gaze.train.custom_trainer import (
    _attach_adaptive_params,
    _build_adaptive_loss,
    _robust_loss,
)


def ccc_loss(logits, labels, eps=1e-8):
    losses = []
    for dim in range(labels.shape[-1]):
        pred = logits[:, dim]
        gold = labels[:, dim]
        pred_mean = pred.mean()
        gold_mean = gold.mean()
        pred_var = pred.var(unbiased=False)
        gold_var = gold.var(unbiased=False)
        cov = ((pred - pred_mean) * (gold - gold_mean)).mean()
        ccc = (2 * cov) / (pred_var + gold_var + (pred_mean - gold_mean).pow(2) + eps)
        losses.append(1 - ccc)
    return torch.stack(losses).mean()


def vad_regression_loss(loss_name, logits, labels, adaptive=None):
    if loss_name == "mse":
        return F.mse_loss(logits, labels)
    if loss_name == "ccc":
        return ccc_loss(logits, labels)
    if loss_name == "mse+ccc":
        return 0.5 * (F.mse_loss(logits, labels) + ccc_loss(logits, labels))
    if loss_name == "robust":
        return _robust_loss(adaptive, logits, labels)
    if loss_name == "robust+ccc":
        return 0.5 * (_robust_loss(adaptive, logits, labels) + ccc_loss(logits, labels))
    raise ValueError(f"Unknown loss name: {loss_name}")


def soft_clip_alignment_loss(z_affect, z_gaze, labels, tau=0.07, sigma=0.05):
    batch_size = labels.shape[0]
    if batch_size < 2:
        return z_affect.sum() * 0.0

    tau = max(float(tau), 1e-6)
    sigma = max(float(sigma), 1e-6)
    labels = labels.detach()
    dist_sq = torch.cdist(labels, labels, p=2).pow(2)
    target = F.softmax(-dist_sq / sigma, dim=-1)

    gaze_to_affect = z_gaze @ z_affect.t() / tau
    affect_to_gaze = z_affect @ z_gaze.t() / tau
    loss_ga = F.kl_div(F.log_softmax(gaze_to_affect, dim=-1), target, reduction="batchmean")
    loss_ag = F.kl_div(F.log_softmax(affect_to_gaze, dim=-1), target, reduction="batchmean")
    return 0.5 * (loss_ga + loss_ag)


class ClipAlignmentTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_name="mse+ccc",
        lambda_align=0.1,
        tau=0.07,
        sigma=0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.lambda_align = float(lambda_align)
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.adaptive = None
        if loss_name in ("robust", "robust+ccc"):
            self.adaptive = _build_adaptive_loss(num_dims=2)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if self.adaptive is not None:
            return _attach_adaptive_params(optimizer, self.adaptive)
        return optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        model_inputs = dict(inputs)
        model_inputs.pop("labels", None)
        outputs = model(**model_inputs)

        logits = outputs.logits
        loss_vad = vad_regression_loss(self.loss_name, logits, labels, adaptive=self.adaptive)
        loss_align = soft_clip_alignment_loss(
            outputs.z_affect,
            outputs.z_gaze,
            labels,
            tau=self.tau,
            sigma=self.sigma,
        )
        loss = loss_vad + self.lambda_align * loss_align

        if return_outputs:
            outputs.loss = loss
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                outputs = model(**inputs)
                loss = None

        if prediction_loss_only:
            return loss, None, None

        logits = outputs.logits.detach()
        if labels is not None:
            labels = labels.detach()
        return loss, logits, labels


def numpy_json_safe(metrics):
    safe = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            value = None
        safe[key] = value
    return safe
