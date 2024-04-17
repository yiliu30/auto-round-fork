# replace `Linear` with `FlexRoundLinear`

import torch
import random
from auto_round.qmodules import QuantizerConfig, FlexRoundLinear, FlexRoundModuleConfig, default_quantizer_config, ada_default_quantizer_config
from auto_round.quantizer import WUniformAffineQuantizer
import pytest
# random.seed(0)
# torch.manual_seed(0)

class TestFlexRoundLinear:
    
    
    def _create_flex_round_linear(cls, in_features=32, out_features=64, quantizer_config=default_quantizer_config):
        config = FlexRoundModuleConfig(weight_config=quantizer_config)
        float_linear = torch.nn.Linear(in_features, out_features)
        flex_round_linear = FlexRoundLinear(float_linear, config=config)
        return float_linear, flex_round_linear
    
    def _get_toy_model(cls, in_features=32):
        class ToyModel(torch.nn.Module):
            def __init__(self, in_features):
                super(ToyModel, self).__init__()
                self.fc1 = torch.nn.Linear(in_features, 64)
                self.fc2 = torch.nn.Linear(64, 32)
            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x
        return ToyModel(in_features=in_features)
    
    def test_quantizer(self):
        tensor = torch.randn(128, 512)
        quantizer = WUniformAffineQuantizer.init_from_tensor(tensor, default_quantizer_config)
        assert len(list(quantizer.parameters())) == 3, "Should have 3 parameters"
    
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
        assert out is not None

    @pytest.mark.parametrize("use_ada",[False, True])
    def test_toy_demo(self, use_ada):
        @torch.no_grad()
        def capture_temp_in_out(iters):
            temp_outs = []
            temp_ins = [] 
            for i in range(iters):
                dummy_input = torch.randn(bs, in_features)
                temp_ins.append(dummy_input)
                out = user_model(dummy_input)
                temp_outs.append(out)
            return temp_outs, temp_ins
        bs = 4
        in_features = 1024
        user_model = self._get_toy_model(in_features)
        train_steps = 100
        temp_out, temp_ins = capture_temp_in_out(train_steps)
        config = FlexRoundModuleConfig(weight_config=QuantizerConfig(n_bits=8, use_ada=use_ada))
        user_model.fc1 = FlexRoundLinear(user_model.fc1, config=config)
        user_model.fc2 = FlexRoundLinear(user_model.fc2, config=config)
        print(user_model)
        params = user_model.fc1.get_trainable_params() + user_model.fc2.get_trainable_params()
        optimizer = torch.optim.Adam(params, lr=1e-1)
        
       
        loss_fn = torch.nn.MSELoss()
        # with torch.no_grad():
        #     float_out = user_model(dummy_input)
        for i in range(train_steps):
            out = user_model(temp_ins[i])
            loss  = loss_fn(temp_out[i], out)
            # loss = torch.sum(temp_out[i] - out)
            if i % 10 ==0:
                print(f"Step {i}, Loss {loss}")
            loss.backward()
            optimizer.step()
            # for name, param in user_model.named_parameters():
            #     print(name, param.grad)
            optimizer.zero_grad()
        # with torch.no_grad():
        #     new_out = user_model(dummy_input)
        #     user_model.fc1.unwrapper()
        #     user_model.fc2.unwrapper()
        #     for i in range(3):
        #         out_after_unwrapper = user_model(dummy_input)
        #         diff = out_after_unwrapper - float_out
        #         # print(diff.min(), diff.max())
        #         # import pdb; pdb.set_trace()
        #         diff2 = out_after_unwrapper - new_out
        #         print(diff2.min(), diff2.max())
        #         assert torch.allclose(out_after_unwrapper, new_out)
        #     assert not torch.allclose(float_out, new_out) #



    def _create_flex_round_linear_ada(cls, in_features=32, out_features=64, quantizer_config=ada_default_quantizer_config):
        config = FlexRoundModuleConfig(weight_config=quantizer_config)
        float_linear = torch.nn.Linear(in_features, out_features)
        flex_round_linear = FlexRoundLinear(float_linear, config=config)
        return float_linear, flex_round_linear

    def test_ada_forward(self):
        in_features = 32
        out_features = 64
        bs = 4
        iters = 3
        float_linear, flex_round_linear = self._create_flex_round_linear_ada(in_features=in_features, out_features=out_features)
        dummy_input = torch.randn(bs, in_features)
        for i in range(iters):
            out = flex_round_linear(dummy_input)
        unwrapped_mod = flex_round_linear.unwrapper()
        print(f"unwrapped_mod: {unwrapped_mod}")
        out = unwrapped_mod(dummy_input)
        assert out is not None
        
        
# pytest  ./test/test_qmodule.py -v -k test_toy_demo
