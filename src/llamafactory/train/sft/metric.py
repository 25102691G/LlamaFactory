# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import logging

import numpy as np
import torch
from transformers.utils import is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_jieba_available, is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


logger = logging.getLogger(__name__)

if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore

try:
    from bertscore import score as bertscore_score  # type: ignore
except ImportError:
    bertscore_score = None


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": [], "bertscore-precision": [], "bertscore-recall": [], "bertscore-f1": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            # Compute BERTScore with detailed logging
            try:
                # Check for empty predictions/labels
                if not pred or not label:
                    logger.warning(f"Empty prediction or label detected. Pred length: {len(pred)}, Label length: {len(label)}")
                    self.score_dict["bertscore-precision"].append(0.0)
                    self.score_dict["bertscore-recall"].append(0.0)
                    self.score_dict["bertscore-f1"].append(0.0)
                    continue
                
                logger.debug(f"Computing BERTScore for sample. Pred length: {len(pred)}, Label length: {len(label)}")
                logger.debug(f"Pred (first 50 chars): {pred[:50]}")
                logger.debug(f"Label (first 50 chars): {label[:50]}")
                
                bertscore_results = bertscore_score(
                    [pred], 
                    [label], 
                    lang="zh", 
                    verbose=False
                )
                
                precision = float(bertscore_results["precision"].mean()) * 100
                recall = float(bertscore_results["recall"].mean()) * 100
                f1 = float(bertscore_results["f1"].mean()) * 100
                
                logger.debug(f"BERTScore raw values - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
                self.score_dict["bertscore-precision"].append(round(precision, 4))
                self.score_dict["bertscore-recall"].append(round(recall, 4))
                self.score_dict["bertscore-f1"].append(round(f1, 4))
                
            except ImportError as e:
                logger.error(f"BERTScore library not available. Please install it with: pip install bert-score. Error: {str(e)}")
                self.score_dict["bertscore-precision"].append(0.0)
                self.score_dict["bertscore-recall"].append(0.0)
                self.score_dict["bertscore-f1"].append(0.0)
            except Exception as e:
                logger.error(f"Failed to compute BERTScore. Error type: {type(e).__name__}, Error: {str(e)}")
                logger.error(f"Prediction: {pred[:100] if pred else 'EMPTY'}")
                logger.error(f"Label: {label[:100] if label else 'EMPTY'}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.score_dict["bertscore-precision"].append(0.0)
                self.score_dict["bertscore-recall"].append(0.0)
                self.score_dict["bertscore-f1"].append(0.0)

        if compute_result:
            return self._dump()
