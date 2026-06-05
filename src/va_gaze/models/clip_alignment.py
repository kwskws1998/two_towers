from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.utils import ModelOutput


@dataclass
class GazeAffectClipOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    z_affect: Optional[torch.FloatTensor] = None
    z_gaze: Optional[torch.FloatTensor] = None
    h_affect: Optional[torch.FloatTensor] = None
    h_gaze: Optional[torch.FloatTensor] = None


class GazeAffectClipModel(nn.Module):
    def __init__(
        self,
        checkpoint,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        projection_dim=256,
        gaze_hidden_dim=256,
        dropout=0.1,
        max_fix_cache_size=20000,
        shuffle_gaze=False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.config = self.encoder.config
        self.tokenizer = tokenizer
        self.hidden_size = self.config.hidden_size
        self.num_labels = 2
        self.shuffle_gaze = shuffle_gaze

        flags = features_used or [1, 1, 1, 1, 1]
        self.feature_indices = [idx for idx, enabled in enumerate(flags) if int(enabled) == 1]
        if not self.feature_indices:
            raise ValueError("features_used must enable at least one ET feature.")

        classifier_dropout = getattr(self.config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(self.config, "seq_classif_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(self.config, "hidden_dropout_prob", dropout)

        self.affect_pre_classifier = nn.Linear(self.hidden_size, self.hidden_size)
        self.affect_dropout = nn.Dropout(classifier_dropout)
        self.vad_head = nn.Linear(self.hidden_size, self.num_labels)

        self.gaze_token_encoder = nn.Sequential(
            nn.Linear(len(self.feature_indices), gaze_hidden_dim),
            nn.LayerNorm(gaze_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gaze_pooler = nn.Sequential(
            nn.Linear(gaze_hidden_dim * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.affect_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, projection_dim),
        )
        self.gaze_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, projection_dim),
        )

        self.fixation_cache = OrderedDict()
        self.max_fix_cache_size = max_fix_cache_size
        self.fp_model = self._load_et2_predictor(et2_checkpoint_path)

    def _load_et2_predictor(self, et2_checkpoint_path):
        try:
            from va_gaze.models.et2_wrapper import FixationsPredictor_2
        except ImportError as exc:
            raise ImportError(
                "Could not import FixationsPredictor_2. Run setup_et_models.py if needed."
            ) from exc

        fp_model = FixationsPredictor_2(
            modelTokenizer=self.tokenizer,
            remap=False,
            checkpoint_path=et2_checkpoint_path,
        )
        if hasattr(fp_model, "model"):
            fp_model.model.eval()
            for param in fp_model.model.parameters():
                param.requires_grad = False
        return fp_model

    @staticmethod
    def _build_cache_key(token_ids_1d, attention_mask_1d):
        valid_len = int(attention_mask_1d.sum().item())
        if valid_len <= 0:
            return tuple(), valid_len
        return tuple(token_ids_1d[:valid_len].tolist()), valid_len

    def _predict_fixations_single(self, token_ids_1d, attention_mask_1d):
        device = token_ids_1d.device
        seq_len = token_ids_1d.shape[0]
        key, valid_len = self._build_cache_key(token_ids_1d, attention_mask_1d)

        if valid_len <= 0:
            return (
                torch.zeros(seq_len, len(self.feature_indices), dtype=torch.float32, device=device),
                torch.zeros(seq_len, dtype=attention_mask_1d.dtype, device=device),
            )

        cached = self.fixation_cache.get(key)
        if cached is None:
            sample_ids = token_ids_1d[:valid_len].unsqueeze(0)
            sample_mask = attention_mask_1d[:valid_len].unsqueeze(0)
            with torch.no_grad():
                fixations, fixation_mask, _, _, _, _ = self.fp_model._compute_mapped_fixations(
                    sample_ids, sample_mask
                )

            fixations = fixations.squeeze(0).float().cpu()
            fixation_mask = fixation_mask.squeeze(0).long().cpu()
            fixations = fixations[:, self.feature_indices]

            if len(self.fixation_cache) >= self.max_fix_cache_size:
                self.fixation_cache.popitem(last=False)
            self.fixation_cache[key] = (fixations, fixation_mask)
        else:
            fixations, fixation_mask = cached
            self.fixation_cache.move_to_end(key)

        fixations = fixations.to(device)
        fixation_mask = fixation_mask.to(device=device, dtype=attention_mask_1d.dtype)

        padded_fixations = torch.zeros(
            seq_len, len(self.feature_indices), dtype=fixations.dtype, device=device
        )
        padded_mask = torch.zeros(seq_len, dtype=attention_mask_1d.dtype, device=device)

        copy_len = min(valid_len, fixations.shape[0], seq_len)
        padded_fixations[:copy_len] = fixations[:copy_len]
        padded_mask[:copy_len] = fixation_mask[:copy_len].to(dtype=attention_mask_1d.dtype)
        return padded_fixations, padded_mask

    def _compute_fixations_batch(self, input_ids, attention_mask):
        batch_fixations = []
        batch_masks = []
        for row_idx in range(input_ids.size(0)):
            row_fix, row_mask = self._predict_fixations_single(
                input_ids[row_idx], attention_mask[row_idx]
            )
            batch_fixations.append(row_fix)
            batch_masks.append(row_mask)
        fixations = torch.stack(batch_fixations, dim=0)
        fixation_attention = torch.stack(batch_masks, dim=0)
        if self.shuffle_gaze and fixations.size(0) > 1:
            perm = torch.randperm(fixations.size(0), device=fixations.device)
            fixations = fixations[perm]
            fixation_attention = fixation_attention[perm]
        return fixations, fixation_attention

    def _pool_gaze(self, fixations, fixation_attention):
        valid = fixation_attention.bool() & fixations.abs().sum(dim=-1).gt(0)
        token_features = self.gaze_token_encoder(fixations)
        valid_f = valid.unsqueeze(-1).to(dtype=token_features.dtype)
        counts = valid_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (token_features * valid_f).sum(dim=1) / counts

        masked = token_features.masked_fill(~valid.unsqueeze(-1), torch.finfo(token_features.dtype).min)
        max_pool = masked.max(dim=1).values
        empty_rows = valid.sum(dim=1).eq(0)
        if empty_rows.any():
            max_pool = max_pool.masked_fill(empty_rows.unsqueeze(-1), 0.0)

        return self.gaze_pooler(torch.cat([mean_pool, max_pool], dim=-1))

    def _pool_affect(self, encoder_outputs):
        pooled = encoder_outputs.last_hidden_state[:, 0, :]
        pooled = self.affect_pre_classifier(pooled)
        pooled = torch.relu(pooled)
        return self.affect_dropout(pooled)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is None:
            raise ValueError("input_ids cannot be None.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        model_device = self.encoder.get_input_embeddings().weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if head_mask is not None:
            encoder_kwargs["head_mask"] = head_mask
        if token_type_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["token_type_ids"] = token_type_ids
        if position_ids is not None and self.config.model_type != "distilbert":
            encoder_kwargs["position_ids"] = position_ids

        encoder_outputs = self.encoder(**encoder_kwargs)
        h_affect = self._pool_affect(encoder_outputs)
        logits = torch.nn.functional.hardsigmoid(3 * self.vad_head(h_affect))
        z_affect = F.normalize(self.affect_projection(h_affect), dim=-1)

        fixations, fixation_attention = self._compute_fixations_batch(input_ids, attention_mask)
        fixations = fixations.to(device=model_device, dtype=h_affect.dtype)
        fixation_attention = fixation_attention.to(device=model_device, dtype=attention_mask.dtype)
        h_gaze = self._pool_gaze(fixations, fixation_attention)
        z_gaze = F.normalize(self.gaze_projection(h_gaze), dim=-1)

        if return_dict is False:
            return logits, z_affect, z_gaze

        return GazeAffectClipOutput(
            loss=None,
            logits=logits,
            z_affect=z_affect,
            z_gaze=z_gaze,
            h_affect=h_affect,
            h_gaze=h_gaze,
        )
