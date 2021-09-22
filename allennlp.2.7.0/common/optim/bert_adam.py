import torch
from pytorch_pretrained_bert.optimization import BertAdam

from allennlp.common import Params, Registrable

import copy
import logging
import re
import math
from typing import Any, Dict, List, Tuple, Union, Optional

from overrides import overrides
import torch
import transformers

from allennlp.common import Params, Registrable, Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer, make_parameter_groups

logger = logging.getLogger(__name__)


ParameterGroupsType = List[Tuple[List[str], Dict[str, Any]]]

@Optimizer.register("bert_adam")
class AdamOptimizer(Optimizer, BertAdam):
    """
    Registered as an `Optimizer` with name "adam".
    """

    def __init__(
        self,
        model_parameters: List[Tuple[str, torch.nn.Parameter]],
        parameter_groups: List[Tuple[List[str], Dict[str, Any]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        super().__init__(
            params=make_parameter_groups(model_parameters, parameter_groups),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )