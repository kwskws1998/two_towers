import os
import re

import numpy as np
import torch
import transformers

try:
    from safetensors.torch import load_file as st_load_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

WINDOW_SIZE = 512
OVERLAP = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _RobertaRegressionModel(torch.nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.roberta = transformers.RobertaModel.from_pretrained(model_name)
        embed_size = 1024 if "large" in model_name else 768
        self.decoder = torch.nn.Linear(embed_size, 5)

    def forward(self, input_ids, attention_mask, predict_mask):
        hidden = self.roberta(input_ids, attention_mask=attention_mask).last_hidden_state
        pred = self.decoder(hidden)
        mask = (predict_mask == 0).unsqueeze(-1).expand_as(pred).to(pred.device)
        pred = pred.masked_fill(mask, -1.0)
        return pred


class FixationsPredictor_2:
    def __init__(self, modelTokenizer, remap=False, checkpoint_path=None, roberta_model_name="roberta-base"):
        self.rm_tokenizer = modelTokenizer
        ckpt = checkpoint_path or os.environ.get("ET2_CHECKPOINT_PATH", "./checkpoints/et_predictor2_seed123")
        self.checkpoint_path = ckpt
        self.roberta_tokenizer = transformers.RobertaTokenizer.from_pretrained(
            roberta_model_name, add_prefix_space=True
        )
        self.model = _RobertaRegressionModel(roberta_model_name).to(device)
        self._load_checkpoint(ckpt)
        self.model.eval()
        print(f"[et2_wrapper] FixationsPredictor_2 loaded: {ckpt}")

    def _load_checkpoint(self, path):
        if os.path.isfile(path):
            return self._load_from_file(path)
        for ext in [".safetensors", ".pt", ".bin"]:
            candidate = path + ext
            if os.path.isfile(candidate):
                return self._load_from_file(candidate)
        raise FileNotFoundError(
            f"[et2_wrapper] checkpoint not found: {path}[.safetensors/.pt]\n"
            "Check ET2_CHECKPOINT_PATH or run setup_et_models.py."
        )

    def _load_from_file(self, path):
        if path.endswith(".safetensors"):
            if not HAS_SAFETENSORS:
                raise ImportError("pip install safetensors")
            state = st_load_file(path, device=str(device))
        else:
            state = torch.load(path, map_location=device)
        self.model.load_state_dict(state, strict=True)
        print(f"[et2_wrapper] weights loaded: {path}")

    def _compute_mapped_fixations(self, input_ids_rm, attention_mask_rm=None):
        if attention_mask_rm is None:
            attention_mask_rm = torch.ones_like(input_ids_rm)

        ids = input_ids_rm[0].cpu().tolist()
        mask = attention_mask_rm[0].cpu().tolist()
        pad_id = self.rm_tokenizer.pad_token_id or 0
        ids_no_pad = [i for i, m in zip(ids, mask) if m == 1 and i != pad_id]
        text = self.rm_tokenizer.decode(ids_no_pad, skip_special_tokens=True)

        word_features, words = self._predict_words(text)
        remapped = self._remap_to_rm_tokens(word_features, words, ids, mask)
        fixations = remapped.unsqueeze(0).to(input_ids_rm.device)
        fix_attn = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(input_ids_rm.device)
        return fixations, fix_attn, None, None, None, None

    def _predict_words(self, text):
        words = self._segment_text(text)
        if not words:
            return np.zeros((0, 5), dtype=np.float32), words

        enc = self.roberta_tokenizer(
            [words],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        token_preds = self._sliding_window_predict(input_ids, attn_mask)
        word_features = self._aggregate_to_words(token_preds, input_ids.squeeze(0))
        return word_features, words

    @staticmethod
    def _is_cjk(ch):
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        )

    def _segment_text(self, text):
        text = (text or "").strip()
        if not text:
            return []
        if any(ch.isspace() for ch in text):
            words = text.split()
            if words:
                return words
        if any(self._is_cjk(ch) for ch in text):
            return [ch for ch in text if not ch.isspace()]
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def _sliding_window_predict(self, input_ids, attn_mask):
        seq_len = input_ids.shape[1]
        if seq_len <= WINDOW_SIZE:
            predict_mask = attn_mask.clone()
            with torch.no_grad():
                pred = self.model(input_ids, attn_mask, predict_mask)
            return pred.squeeze(0).cpu().numpy()

        preds = np.zeros((seq_len, 5), dtype=np.float32)
        weights = np.zeros(seq_len, dtype=np.float32)
        stride = WINDOW_SIZE - OVERLAP
        start = 0

        while start < seq_len:
            end = min(start + WINDOW_SIZE, seq_len)
            ids_win = input_ids[:, start:end]
            mask_win = attn_mask[:, start:end]
            win_len = end - start

            linear_w = np.ones(win_len, dtype=np.float32)
            if start > 0:
                rl = min(OVERLAP, win_len)
                linear_w[:rl] = np.linspace(0, 1, rl)
            if end < seq_len:
                rl = min(OVERLAP, win_len)
                linear_w[-rl:] = np.linspace(1, 0, rl)

            with torch.no_grad():
                pred_win = self.model(ids_win, mask_win, mask_win.clone())
            pred_np = pred_win.squeeze(0).cpu().numpy()

            for fi in range(5):
                preds[start:end, fi] += pred_np[:, fi] * linear_w
            weights[start:end] += linear_w

            if end == seq_len:
                break
            start += stride

        nz = weights > 0
        preds[nz] /= weights[nz, None]
        return preds

    def _aggregate_to_words(self, token_preds, input_ids_1d):
        tokens = [self.roberta_tokenizer.convert_ids_to_tokens(int(i)) for i in input_ids_1d]
        word_feats = []
        seen_first_word = False
        for idx, tok in enumerate(tokens):
            if tok in ("<s>", "</s>", "<pad>"):
                continue
            if tok.startswith("Ġ") or not seen_first_word:
                pred = np.clip(token_preds[idx], 0, None)
                word_feats.append(pred)
                seen_first_word = True
        if word_feats:
            return np.array(word_feats, dtype=np.float32)
        return np.zeros((0, 5), dtype=np.float32)

    def _remap_to_rm_tokens(self, word_features, words, rm_ids, rm_mask):
        seq_len = len(rm_ids)
        output = torch.zeros(seq_len, 5, dtype=torch.float32)
        if len(word_features) == 0 or len(words) == 0:
            return output

        rm_tokens = self.rm_tokenizer.convert_ids_to_tokens(rm_ids)
        word_to_rm = _align_words_to_rm_tokens(words, rm_tokens, self.rm_tokenizer)

        n_words = min(len(words), len(word_features))
        for w_idx in range(n_words):
            if w_idx >= len(word_to_rm):
                break
            indices = word_to_rm[w_idx]
            if not indices:
                continue
            first = indices[0]
            if first < seq_len and rm_mask[first] == 1:
                output[first] = torch.tensor(word_features[w_idx], dtype=torch.float32)
        return output


def _align_words_to_rm_tokens(words, rm_tokens, rm_tokenizer):
    special_ids = set(rm_tokenizer.all_special_ids)
    word_to_indices = []
    tok_idx = 0

    for word in words:
        indices = []
        chars_remaining = len(word)
        while tok_idx < len(rm_tokens) and chars_remaining > 0:
            tok = rm_tokens[tok_idx]
            tok_id = rm_tokenizer.convert_tokens_to_ids(tok)
            if tok_id in special_ids:
                tok_idx += 1
                continue

            tok_clean = tok.lstrip("Ġ▁ ")
            indices.append(tok_idx)
            chars_remaining -= len(tok_clean)
            tok_idx += 1
        word_to_indices.append(indices)
    return word_to_indices
