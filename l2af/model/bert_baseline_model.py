import logging
from typing import Any, Dict, List
import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Auc
from allennlp.nn.util import get_text_field_mask

from l2af.nn.layers import AttnPooling

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("l2af_bert_baseline")
class L2AFBertBaseline(Model):
    """
    BERT baseline model

    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 attnpool: AttnPooling,
                 output_ffl: FeedForward,
                 initializer: InitializerApplicator,
                 dropout: float = 0.3,
                 ) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder

        self._variational_dropout = InputVariationalDropout(dropout)
        self._attn_pool = attnpool
        self._output_ffl = output_ffl

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._auc = Auc()
        self._loss = torch.nn.BCELoss()
        initializer(self)

    def forward(self,  # type: ignore
                combined_source: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # pylint: disable=arguments-differ
        """

        :param combined_source:
        :param label:
        :param metadata:
        :return:
        """
        embedded_source = self._text_field_embedder(combined_source)  # B * T * H
        source_mask = get_text_field_mask(combined_source)  # B * T
        embedded_source = self._variational_dropout(embedded_source)

        pooled = self._attn_pool(embedded_source, source_mask)  # B * H
        choice_score = self._output_ffl(pooled)  # B * 1

        output = torch.sigmoid(choice_score).squeeze(-1)  # B

        output_dict = {"label_logits": choice_score.squeeze(-1), "label_probs": output,
                       "metadata": metadata}

        if label is not None:
            label = label.long().view(-1)
            loss = self._loss(output, label.float())
            self._auc(output, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'auc': self._auc.get_metric(reset),
        }
