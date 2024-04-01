# replace `Linear` with `FlexRoundLinear`

## Test
import torch
from auto_round.qmodules import QuantizerConfig, FlexRoundLinear, FlexRoundModuleConfig

weight_quantizer_config = QuantizerConfig(n_bits=8)
flex_round_linear_config = FlexRoundModuleConfig(weight_config = weight_quantizer_config)
print(flex_round_linear_config)
in_features = 32
out_features = 64
float_linear = torch.nn.Linear(in_features, out_features)
flex_round_linear = FlexRoundLinear(float_linear, config=flex_round_linear_config)

trainable_params = flex_round_linear.get_trainable_params()
print(trainable_params)