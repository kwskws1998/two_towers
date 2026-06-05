from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, DistilBertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaClassificationHead,
    RobertaForSequenceClassification,
)
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig


class DistilBertForSequenceClassificationSig(DistilBertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = lambda x: torch.nn.functional.hardsigmoid(3 * x)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        ret = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ret.logits = self.sigmoid(ret.logits)
        return ret


class RobertaForSequenceClassificationSig(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.sigmoid = lambda x: torch.nn.functional.hardsigmoid(3 * x)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        ret = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ret.logits = self.sigmoid(ret.logits)
        return ret


class XLMRobertaForSequenceClassificationSig(RobertaForSequenceClassificationSig):
    config_class = XLMRobertaConfig


class GazeConcatForSequenceRegression(nn.Module):
    def __init__(
        self,
        checkpoint,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        fp_dropout=(0.0, 0.3),
        max_fix_cache_size=20000,
        load_fixation_model=True,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.config = self.encoder.config
        self.tokenizer = tokenizer
        self.hidden_size = self.config.hidden_size
        self.num_labels = 2

        flags = features_used or [1, 1, 1, 1, 1]
        self.feature_indices = [idx for idx, enabled in enumerate(flags) if int(enabled) == 1]
        if not self.feature_indices:
            raise ValueError("features_used must enable at least one ET feature.")

        p_1, p_2 = fp_dropout
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(len(self.feature_indices), 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, self.hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(self.hidden_size)

        classifier_dropout = getattr(self.config, "classifier_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(self.config, "seq_classif_dropout", None)
        if classifier_dropout is None:
            classifier_dropout = getattr(self.config, "hidden_dropout_prob", 0.1)

        self.pre_classifier = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.sigmoid = torch.nn.functional.hardsigmoid

        self.eye_start = nn.Parameter(torch.zeros(self.hidden_size))
        self.eye_end = nn.Parameter(torch.zeros(self.hidden_size))
        self.fixation_cache = OrderedDict()
        self.max_fix_cache_size = max_fix_cache_size
        self.fp_model = self._load_et2_predictor(et2_checkpoint_path) if load_fixation_model else None

    def _load_et2_predictor(self, et2_checkpoint_path):
        try:
            from va_gaze.models.et2_wrapper import FixationsPredictor_2
        except ImportError as exc:
            raise ImportError(
                "Could not import FixationsPredictor_2. Make sure et2_wrapper.py exists and run setup_et_models.py if needed."
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
        return torch.stack(batch_fixations, dim=0), torch.stack(batch_masks, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if input_ids is None:
            raise ValueError("input_ids cannot be None.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embed_layer = self.encoder.get_input_embeddings()
        model_device = embed_layer.weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        text_embeddings = embed_layer(input_ids)
        fixations, fixation_attention = self._compute_fixations_batch(input_ids, attention_mask)
        fixations = fixations.to(device=model_device, dtype=text_embeddings.dtype)
        fixation_attention = fixation_attention.to(device=model_device, dtype=attention_mask.dtype)

        fixations_projected = self.fixations_embedding_projector(fixations)
        fixations_projected = self.norm_layer_fix(fixations_projected)

        batch_size = input_ids.size(0)
        eye_start_embed = self.eye_start.to(device=model_device, dtype=text_embeddings.dtype).view(1, 1, -1)
        eye_end_embed = self.eye_end.to(device=model_device, dtype=text_embeddings.dtype).view(1, 1, -1)
        eye_start_embed = eye_start_embed.expand(batch_size, -1, -1)
        eye_end_embed = eye_end_embed.expand(batch_size, -1, -1)
        separator_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=model_device)

        inputs_embeds = torch.cat(
            (eye_start_embed, fixations_projected, eye_end_embed, text_embeddings), dim=1
        )
        extended_attention_mask = torch.cat(
            (separator_mask, fixation_attention, separator_mask, attention_mask), dim=1
        )

        encoder_kwargs = {
            "input_ids": None,
            "attention_mask": extended_attention_mask,
            "inputs_embeds": inputs_embeds,
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
        cls_position = fixations_projected.shape[1] + 2
        pooled_output = encoder_outputs.last_hidden_state[:, cls_position, :]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = torch.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)

        if return_dict is False:
            return (logits,)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GazeAddForSequenceRegression(GazeConcatForSequenceRegression):
    def __init__(
        self,
        checkpoint,
        tokenizer,
        et2_checkpoint_path=None,
        features_used=None,
        fp_dropout=(0.0, 0.3),
        max_fix_cache_size=20000,
        gaze_add_scale=0.05,
        train_gaze_add_scale=False,
    ):
        skip_fixed_zero_gaze = not train_gaze_add_scale and float(gaze_add_scale) == 0.0
        super().__init__(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            et2_checkpoint_path=et2_checkpoint_path,
            features_used=features_used,
            fp_dropout=fp_dropout,
            max_fix_cache_size=max_fix_cache_size,
            load_fixation_model=not skip_fixed_zero_gaze,
        )
        self.skip_fixed_zero_gaze = skip_fixed_zero_gaze
        gaze_add_scale = torch.tensor(float(gaze_add_scale))
        if train_gaze_add_scale:
            self.gaze_add_scale = nn.Parameter(gaze_add_scale)
        else:
            self.register_buffer("gaze_add_scale", gaze_add_scale)
        self.sigmoid = lambda x: torch.nn.functional.hardsigmoid(3 * x)
        if self.config.model_type != "distilbert":
            self.config.num_labels = self.num_labels
            self.roberta_classifier = RobertaClassificationHead(self.config)
            self._init_roberta_classifier()

    def _init_roberta_classifier(self):
        initializer_range = getattr(self.config, "initializer_range", 0.02)
        for module in self.roberta_classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if input_ids is None:
            raise ValueError("input_ids cannot be None.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embed_layer = self.encoder.get_input_embeddings()
        model_device = embed_layer.weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)

        text_embeddings = embed_layer(input_ids)
        if self.skip_fixed_zero_gaze:
            inputs_embeds = text_embeddings
        else:
            fixations, _ = self._compute_fixations_batch(input_ids, attention_mask)
            fixations = fixations.to(device=model_device, dtype=text_embeddings.dtype)

            fixations_projected = self.fixations_embedding_projector(fixations)
            fixations_projected = self.norm_layer_fix(fixations_projected)
            gaze_present = fixations.abs().sum(dim=-1, keepdim=True).gt(0).to(dtype=text_embeddings.dtype)
            fixations_projected = fixations_projected * gaze_present
            inputs_embeds = text_embeddings + self.gaze_add_scale * fixations_projected

        encoder_kwargs = {
            "input_ids": None,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
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
        if self.config.model_type == "distilbert":
            pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
            pooled_output = self.pre_classifier(pooled_output)
            pooled_output = torch.relu(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        else:
            logits = self.roberta_classifier(encoder_outputs.last_hidden_state)
        logits = self.sigmoid(logits)

        if return_dict is False:
            return (logits,)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
