import numpy as np
import robust_loss_pytorch
import torch
from transformers import Trainer

from va_gaze.eval.oof_reports import pearsonr


def _pop_labels(inputs):
    labels = inputs["labels"]
    model_inputs = dict(inputs)
    model_inputs.pop("labels", None)
    return labels, model_inputs


def _ccc_loss(logits, labels):
    logits_v = logits[:, 0]
    logits_a = logits[:, 1]
    labels_v = labels[:, 0]
    labels_a = labels[:, 1]

    num_v = 2 * pearsonr(logits_v, labels_v) * torch.std(logits_v) * torch.std(labels_v)
    den_v = torch.var(logits_v) + torch.var(labels_v) + torch.square(torch.mean(logits_v) - torch.mean(labels_v))
    ccc_v = num_v / den_v
    loss_v = 1 - ccc_v

    num_a = 2 * pearsonr(logits_a, labels_a) * torch.std(logits_a) * torch.std(labels_a)
    den_a = torch.var(logits_a) + torch.var(labels_a) + torch.square(torch.mean(logits_a) - torch.mean(labels_a))
    ccc_a = num_a / den_a
    loss_a = 1 - ccc_a

    return 0.5 * loss_v + 0.5 * loss_a


def _build_adaptive_loss(num_dims=2):
    adaptive_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        num_dims=num_dims,
        float_dtype=np.float32,
        device=adaptive_device,
    )


def _attach_adaptive_params(optimizer, adaptive):
    adaptive_params = [p for p in adaptive.parameters() if p.requires_grad]
    if not adaptive_params:
        return optimizer

    existing_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    new_params = [param for param in adaptive_params if id(param) not in existing_param_ids]
    if new_params:
        optimizer.add_param_group({"params": new_params, "weight_decay": 0.0})
    return optimizer


def _robust_loss(adaptive, logits, labels):
    if next(adaptive.parameters()).device != logits.device:
        adaptive.to(logits.device)
    residual = labels - logits
    return torch.mean(adaptive.lossfun(residual))


class CustomTrainerMSE(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class CustomTrainerCCC(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = _ccc_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerMSE_CCC(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        mse_loss = torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1))
        ccc_loss = _ccc_loss(logits, labels)
        loss = 0.5 * (mse_loss + ccc_loss)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerRobust(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = _build_adaptive_loss(num_dims=2)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        return _attach_adaptive_params(optimizer, self.adaptive)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        loss = _robust_loss(self.adaptive, logits, labels)
        return (loss, outputs) if return_outputs else loss


class CustomTrainerRobustCCC(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = _build_adaptive_loss(num_dims=2)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        return _attach_adaptive_params(optimizer, self.adaptive)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels, model_inputs = _pop_labels(inputs)
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        robust = _robust_loss(self.adaptive, logits, labels)
        ccc = _ccc_loss(logits, labels)
        loss = 0.5 * (robust + ccc)
        return (loss, outputs) if return_outputs else loss
