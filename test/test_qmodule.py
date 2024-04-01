# replace `Linear` with `FlexRoundLinear`

import torch
from auto_round.qmodules import QuantizerConfig, FlexRoundLinear, FlexRoundModuleConfig, default_quantizer_config

class TestFlexRoundLinear:
    
    
    def _create_flex_round_linear(cls, in_features=32, out_features=64, quantizer_config=default_quantizer_config):
        config = FlexRoundModuleConfig(weight_config=quantizer_config)
        float_linear = torch.nn.Linear(in_features, out_features)
        flex_round_linear = FlexRoundLinear(float_linear, config=config)
        return float_linear, flex_round_linear
    
    def test_all(self):
        in_features = 32
        out_features = 64
        float_linear, flex_round_linear = self._create_flex_round_linear(in_features=in_features, out_features=out_features)

        trainable_params = flex_round_linear.get_trainable_params()
        assert len(trainable_params) == 3
        assert trainable_params[0].numel() == 1 # s1
        assert trainable_params[1].shape == float_linear.weight.shape # s2
        assert len(trainable_params[2].shape) == len(float_linear.weight.shape) # s3 [out_feature, 1]
        assert trainable_params[2].shape[0] == out_features # s3
        
    def test_unwrapper(self):
        float_linear, flex_round_linear = self._create_flex_round_linear()
        flex_round_linear.unwrapper()
        assert flex_round_linear.inference_mode is True
        
        
    def test_forward(self):
        in_features = 32
        out_features = 64
        bs = 4
        iters = 3
        float_linear, flex_round_linear = self._create_flex_round_linear(in_features=in_features, out_features=out_features)
        dummy_input = torch.randn(bs, in_features)
        for i in range(iters):
            out = flex_round_linear(dummy_input)
        flex_round_linear.unwrapper()
        out = flex_round_linear(dummy_input)
        
        