from typing import Any, Callable, Dict, List, Optional, Union
import collections
from collections import defaultdict
import copy
import os
import re
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
#合并多个参数，就是深度合并多个字典。-> dict表示最后返回一个 dict
def merge_dicts(*dicts: dict) -> dict:
    """Recursive dict merge.
    Instead of updating only top-level keys,
    ``merge_dicts`` recurses down into dicts nested
    to an arbitrary depth, updating keys.

    Args:
        *dicts: several dictionaries to merge

    Returns:
        dict: deep-merged dictionary
    """
    assert len(dicts) > 1#至少需要两个字典
    #深复制dicts[0]，后面所有字典都往这个基础上加
    dict_ = copy.deepcopy(dicts[0])
    #遍历除第一个之外的所有字典
    for merge_dict in dicts[1:]:
        merge_dict = merge_dict or {}#防止字典为空出现报错
        #开始遍历字典中的key部分
        for k in merge_dict:
            if (
                k in dict_#看这个key是否已经存在
                and isinstance(dict_[k], dict)#然后再看这个key对应的value是不是字典。
                and isinstance(merge_dict[k], collections.Mapping)#再看新value是不是字典
            ):
                dict_[k] = merge_dicts(dict_[k], merge_dict[k])
            else:
                dict_[k] = merge_dict[k]#若不是两个dict，就用后面的值直接覆盖

    return dict_

#处理模型参数，给模型参数按照规则分组。（给模型不同部分的参数起名，并明确每一部分参数的优化规则）
def process_model_params(
    model:nn.Module,
    layerwise_params: Dict[str, dict] = None,#参数名字匹配规则（wise是层的意思，layerwise就是按照层）
    no_bias_weight_decay: bool = True,
    lr_scaling: float = 1.0,#学习率缩放
) -> List[Union[torch.nn.Parameter, dict]]:
    """Gains model parameters for ``torch.optim.Optimizer``.

    Args:
        model (torch.nn.Module): Model to process
        layerwise_params (Dict): Order-sensitive dict where
            each key is regex pattern and values are layer-wise options
            for layers matching with a pattern
        no_bias_weight_decay (bool): If true, removes weight_decay
            for all ``bias`` parameters in the model
        lr_scaling (float): layer-wise learning rate scaling,
            if 1.0, learning rates will not be scaled

    Returns:
        iterable: parameters for an optimizer

    Example::

    """
    params = list(model.named_parameters())#获取所有模型参数，注意，这里得到的参数都是配有对应的名字的
    layerwise_params = layerwise_params or collections.OrderedDict()#若没有传，则创建空字典

    model_params = []
    for name, parameters in params:#开始遍历参数
        options = {}#保存这个参数的特殊设置
        #匹配层规则，把对应层的规则加上去
        for pattern, pattern_options in layerwise_params.items():
            if re.match(pattern, name) is not None:#正则匹配，若pattern跟某一个name匹配成功
                # all new LR rules write on top of the old ones
                options = merge_dicts(options, pattern_options)

        # no bias decay from https://arxiv.org/abs/1812.01187
        if no_bias_weight_decay and name.endswith("bias"):
            options["weight_decay"] = 0.0

        # lr linear scaling from https://arxiv.org/pdf/1706.02677.pdf
        if "lr" in options:
            options["lr"] *= lr_scaling

        model_params.append({"params": parameters, **options})

    return model_params

#优化器增强算法
class Lookahead(Optimizer):
    """Implements Lookahead algorithm.

    It has been proposed in `Lookahead Optimizer: k steps forward,
    1 step back`_.

    Adapted from:
    https://github.com/alphadl/lookahead.pytorch (MIT License)

    .. _`Lookahead Optimizer\: k steps forward, 1 step back`:
        https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        """@TODO: Docs. Contribution is welcome."""
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        """@TODO: Docs. Contribution is welcome."""
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        """@TODO: Docs. Contribution is welcome."""
        for group in self.param_groups:
            self.update(group)

    def step(self, closure: Optional[Callable] = None):
        """Makes optimizer step.

        Args:
            closure (callable, optional): A closure that reevaluates
                the model and returns the loss.

        Returns:
            computed loss
        """
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        """@TODO: Docs. Contribution is welcome."""
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        """@TODO: Docs. Contribution is welcome."""
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        """@TODO: Docs. Contribution is welcome."""
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)

    @classmethod
    def get_from_params(
        cls, params: Dict, base_optimizer_params: Dict = None, **kwargs,
    ) -> "Lookahead":
        """@TODO: Docs. Contribution is welcome."""
        from catalyst.registry import OPTIMIZERS

        base_optimizer = OPTIMIZERS.get_from_params(
            params=params, **base_optimizer_params
        )
        optimizer = cls(optimizer=base_optimizer, **kwargs)
        return optimizer