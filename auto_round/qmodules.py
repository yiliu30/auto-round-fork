import torch
from dataclasses import dataclass
from typing import List

import torch.nn.functional as F
import logging 

logger = logging.getLogger(__name__)
from .quantizer import WUniformAffineQuantizer,QuantizerConfig, default_quantizer_config, AdaRoundQuantizer, ada_default_quantizer_config


@dataclass
class FlexRoundModuleConfig:
    weight_config: QuantizerConfig
    input_config: QuantizerConfig = None
    output_config: QuantizerConfig = None




class FlexRoundLinear(torch.nn.Module):
    def __init__(self,  orig_layer: torch.nn.Linear, enable_minmax_tuning=False, config: FlexRoundModuleConfig=None):
        assert enable_minmax_tuning is False, "`enable_minmax_tuning` is a placeholder only."
        super(FlexRoundLinear, self).__init__()
        logger.info(f"Create flexRoundLinear with: {config}")
        self.config = config
        self._orig_layer = orig_layer
        if self.config.weight_config:
            if self.config.weight_config.use_ada:
                self.weight_quantizer = AdaRoundQuantizer.init_from_tensor(self._orig_layer.weight, self.config.weight_config)
            else:
                self.weight_quantizer = WUniformAffineQuantizer.init_from_tensor(self._orig_layer.weight, self.config.weight_config)
        self.inference_mode = False
        
    def forward(self, input: torch.Tensor):
        if self.config.weight_config:
            weight = self.weight_quantizer(self._orig_layer.weight)
            bias = self._orig_layer.bias
        else:
            raise ValueError("Neither inference mode nor quantization is enabled.")
        
        out = F.linear(input, weight, bias)
        return out

    def unwrapper(self):
        # Unrapper the module for inference
        # 1. use the last delata to update the weight and assign to `self.weight`
        # 2. set `self.inference_mode` to True 
        # * Note it not use the best deltas
        
        with torch.no_grad():
            final_weight = self.weight_quantizer(self._orig_layer.weight)
            self._orig_layer.weight.data.copy_(final_weight)
            self._orig_layer.weight.requires_grad_ = False
            self.inference_mode = True
            logger.info("Set FlexRoundLieaner to inference mode.")
            return self._orig_layer
    
    def get_trainable_params(self) -> List[torch.Tensor]:
        if self.config.weight_config:
            return list(self.weight_quantizer.parameters())
        else:
            return []


