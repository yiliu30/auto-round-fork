import dataclasses
from typing import List, Optional, Tuple

import awq.models.auto
import pytest
import torch
import transformers.models
from accelerate import init_empty_weights
from torch._subclasses.fake_tensor import FakeTensorMode
import transformers
import awq
import awq.models
import awq.quantize
import awq.quantize.scale
import awq.utils
import awq.utils.utils
import awq.utils.utils as awq_utils

from auto_round.utils import logger

def _tensor_bytes(t: torch.Tensor):
    logger.debug(f"Tensor shape: {t.shape}, dtype: {t.dtype}, device: {t.device}")
    return t.numel() * t.element_size()

def _any_type_input_size(input):
    if isinstance(input, torch.Tensor):
        return _tensor_bytes(input)
    elif isinstance(input, tuple):
        return sum(_any_type_input_size(i) for i in input) or 0
    elif isinstance(input, list):
        return sum(_any_type_input_size(i) for i in input) or 0
    elif isinstance(input, dict):
        return sum(_any_type_input_size(i) for i in input.values()) or 0
    else:
        logger.info(f"Unsupported type: {type(input)}")
        return 0

def _debug_input(*args, **kwargs):
    final_size = sum(_any_type_input_size(i) for i in args) + sum(_any_type_input_size(i) for i in kwargs.values())
    logger.info(f"Input size: {final_size}, need mem: {final_size / 1024 / 1024} MB")
    return final_size

def assert_same(
    a: Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]],
    b: Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]],
):
    assert len(a) == len(b), f"len: {len(a)} != {len(b)}"
    for i in range(len(a)):
        assert type(a[i]) == type(b[i]), f"type: {type(a[i])} != {type(b[i])}"
        if isinstance(a[i], torch.Tensor):
            torch.testing.assert_allclose(a[i], b[i])
        elif isinstance(a[i], tuple):
            assert_same(a[i], b[i])
        else:
            raise ValueError(f"Unsupported type: {type(a[i])}")
    print("Same!")


def update_best_scale(block: torch.nn.Module):
    for _, mod in block.named_modules():
        if isinstance(mod, ScaleMod):
            mod.update_best_scale()



class ScaleModV1(torch.nn.Module):
    def __init__(self, shape: int):
        super().__init__()
        self.scales = torch.nn.Parameter(torch.ones(shape), requires_grad=True)
        self._best_scales = self.scales.clone().detach().requires_grad_(False)
    
    def update_best_scale(self):
        self._best_scales.copy_(self.scales.data)
    
    def get_best_scale(self):
        return self._best_scales.detach().cpu()
    
    def forward(self, x):
        return self.scales.view(1, -1)

    def __repr__(self):
        return f"ScaleModV1(shape={self.scales.shape})"
    
class ScaleModV2(torch.nn.Module):
    def __init__(self, shape: int):
        super().__init__()
        # clip to 0~1
        # lr, 1.0/steps -0.5, 0.5
        self.scales1 = torch.nn.Parameter(torch.ones(shape), requires_grad=True)
        self.scales2 = torch.nn.Parameter(torch.ones(shape), requires_grad=True)
        self._best_scales = self._get_final_scale().detach().requires_grad_(False)

    def _get_final_scale(self):
        s1 = self.scales1
        s2 = self.scales2
        s2 = torch.where(s2 == 0, torch.ones_like(s2), s2)
        return s1/s2
    
    @torch.no_grad()
    def update_best_scale(self):
        self._best_scales.copy_(self._get_final_scale())
    
    @torch.no_grad()
    def get_best_scale(self):
        return self._best_scales.detach().cpu()
    
    def forward(self, x):
        final_scale = self._get_final_scale()
        return final_scale.view(1, -1)

    def __repr__(self):
        return f"ScaleModV2(scale1={self.scales1.shape}, scale2={self.scales2.shape})"



class ScaleModV3(torch.nn.Module):
    def __init__(self, shape: int):
        super().__init__()
        # clip to 0~1
        # lr, 1.0/steps -0.5, 0.5
        # 200 steps, 1/200, (0 + 1/200) * 200 /2
        self.scales1 = torch.nn.Parameter(torch.ones(shape) * 0.5, requires_grad=True) 
        self.scales2 = torch.nn.Parameter(torch.ones(shape)  * 0.5, requires_grad=True)
        self._best_scales = self._get_final_scale().detach().requires_grad_(False)

    def _get_final_scale(self):
        s1 = self.scales1.clamp(0.0, 1.0)
        s2 = self.scales2.clamp(0.0, 1.0)
        s2 = torch.where(s2 == 0, torch.ones_like(s2), s2)
        return s1/s2
    
    @torch.no_grad()
    def update_best_scale(self):
        self._best_scales.copy_(self._get_final_scale())
    
    @torch.no_grad()
    def get_best_scale(self):
        return self._best_scales.detach().cpu()
    
    def forward(self, x):
        final_scale = self._get_final_scale()
        return final_scale.view(1, -1)

    def __repr__(self):
        return f"ScaleModV3(scale1={self.scales1.shape}, scale2={self.scales2.shape})"



import os


scale_mod_version = os.environ.get("ScaleMod", "1")

if scale_mod_version == "1":
    ScaleMod = ScaleModV1
elif scale_mod_version == "2":
    ScaleMod = ScaleModV2
elif scale_mod_version == "3":
    ScaleMod = ScaleModV3
else:
    raise ValueError(f"Unsupported ScaleMod version: {scale_mod_version}")
print(f"Using ScaleMod version: {scale_mod_version}")

class DivLinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Module, scale_mod: ScaleMod):
        super().__init__()
        self.linear = linear
        freeze_mod_with_filter_(self.linear)
        self.scale_mod = scale_mod

    def forward(self, x):
        x = self.linear(x)
        x = x / self.scale_mod(x)
        return x


class MulLinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, scale_mod: ScaleMod):
        super().__init__()
        self.linear = linear
        freeze_mod_with_filter_(self.linear)
        self.scale_mod = scale_mod

    def forward(self, x):
        x = x * self.scale_mod(x)
        x = self.linear(x)
        return x


@dataclasses.dataclass
class PairInfo:
    prev_op_name: str
    layer_names: List[str]


def _create_layer_info(layers_info):
    module_layer_info = []
    for pair in layers_info:
        prev_op_name = pair["prev_op_name"]
        layer_names = pair["layer_names"]
        module_layer_info.append(PairInfo(prev_op_name, layer_names))
    return module_layer_info


def freeze_mod_(mod: torch.nn.Module):
    for p in mod.parameters():
        p.requires_grad = False


def freeze_mod_with_filter_(mod, allowed_filter=(MulLinear, DivLinear, ScaleMod)):
    for name, module in mod.named_modules():
        if isinstance(module, allowed_filter):
            continue
        freeze_mod_(module)


def get_tranable_params(mod: torch.nn.Module):
    return [p for p in mod.parameters() if p.requires_grad]


def replace_(llama_decoder_layer, module_pairs_info: List[PairInfo]) -> None:
    device = next(llama_decoder_layer.parameters()).device
    for pair_info in module_pairs_info:
        prev_op_name = pair_info.prev_op_name
        layer_names = pair_info.layer_names
        first_layer = awq_utils.get_module_by_name_suffix(llama_decoder_layer, layer_names[0])
        scale_shape = first_layer.in_features  # or one_layer.linear.in_features
        scale_mod = ScaleMod(scale_shape).to(device)
        for layer_name in layer_names:
            layer = awq_utils.get_module_by_name_suffix(llama_decoder_layer, layer_name)
            awq_utils.set_module_name(llama_decoder_layer, layer_name, MulLinear(layer, scale_mod))
        _pre_op = awq_utils.get_module_by_name_suffix(llama_decoder_layer, prev_op_name)
        awq_utils.set_module_name(llama_decoder_layer, prev_op_name, DivLinear(_pre_op, scale_mod))


def _revert_replace(llama_decoder_layer, module_pairs_info: List[PairInfo]):
    scale_lst = []
    for pair_info in module_pairs_info[::-1]:
        prev_op_name = pair_info.prev_op_name
        layer_names = pair_info.layer_names
        div_linear = awq_utils.get_module_by_name_suffix(llama_decoder_layer, prev_op_name)
        awq_utils.set_module_name(llama_decoder_layer, prev_op_name, div_linear.linear)
        scale_lst.append((prev_op_name, layer_names, div_linear.scale_mod.get_best_scale()))
        for _layer_name in layer_names:
            mul_layer = awq_utils.get_module_by_name_suffix(llama_decoder_layer, _layer_name)
            awq_utils.set_module_name(llama_decoder_layer, _layer_name, mul_layer.linear)
    return llama_decoder_layer, scale_lst[::-1]


def absorb_mul_(llama_decoder_layer, module_pairs_info):
    llama_decoder_layer, scale_lst = _revert_replace(llama_decoder_layer, module_pairs_info)
    awq.quantize.scale.apply_scale(llama_decoder_layer, scale_lst)


def _get_module_info(block, model_type):
    awq_model_cls = awq.models.auto.AWQ_CAUSAL_LM_MODEL_MAP[model_type]
    fake_input_feat = awq_model_cls.fake_input_feat()
    layers_info = awq_model_cls.get_layers_for_scaling(block, fake_input_feat, module_kwargs=None)
    module_pairs_info = _create_layer_info(layers_info)
    return module_pairs_info

def api_replace_(block, model_type):
    module_pairs_info = _get_module_info(block, model_type)
    block._module_pairs_info = module_pairs_info
    replace_(block, module_pairs_info=module_pairs_info)

def api_absorb_mul_(block):
    module_pairs_info = block._module_pairs_info
    absorb_mul_(block, module_pairs_info=module_pairs_info)
    del block._module_pairs_info

class Test:
    def test_replace(self):
        import torch
        import transformers

        model_name = "/data5/yliu7/Llama-2-7b-chat-hf/"
        from transformers.models.llama import modeling_llama

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        llama_decoder_layer: modeling_llama.LlamaDecoderLayer = model.model.layers[0]
        batch, seq_len, embed_dim = 2, 10, 4096
        position_ids = torch.arange(seq_len).unsqueeze(0)
        hidden_states = torch.randn(batch, seq_len, embed_dim)
        output_ref = llama_decoder_layer(hidden_states, position_ids=position_ids)

        def replace_decoder(llama_decoder_layer: modeling_llama.LlamaDecoderLayer):
            self_attn = llama_decoder_layer.self_attn
            mlp = llama_decoder_layer.mlp
            act1 = ScaleMod(self_attn.q_proj.in_features)
            self_attn.q_proj = MulLinear(self_attn.q_proj, act1)
            self_attn.k_proj = MulLinear(self_attn.k_proj, act1)
            self_attn.v_proj = MulLinear(self_attn.v_proj, act1)

            act2 = ScaleMod(self_attn.o_proj.in_features)
            self_attn.o_proj = MulLinear(self_attn.o_proj, act2)

            act3 = ScaleMod(mlp.gate_proj.in_features)

            mlp.gate_proj = MulLinear(mlp.gate_proj, act3)
            mlp.up_proj = MulLinear(mlp.up_proj, act3)

            act4 = ScaleMod(mlp.down_proj.in_features)
            mlp.down_proj = MulLinear(mlp.down_proj, act4)

            llama_decoder_layer.self_attn = self_attn
            llama_decoder_layer.mlp = mlp
            return llama_decoder_layer

        llama_decoder_layer_with_scale = replace_decoder(llama_decoder_layer)
        output = llama_decoder_layer_with_scale(hidden_states, position_ids=position_ids)
        assert_same(output_ref, output)

    def _test_replace_ref(self):
        import torch
        import transformers

        model_name = "/data5/yliu7/Llama-2-7b-chat-hf/"
        from transformers.models.llama import modeling_llama

        # with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        llama_decoder_layer: modeling_llama.LlamaDecoderLayer = model.model.layers[0]
        model.eval()
        fake_input_feat = awq.models.llama.LlamaAWQForCausalLM.fake_input_feat()
        layers_info = awq.models.llama.LlamaAWQForCausalLM.get_layers_for_scaling(
            llama_decoder_layer, fake_input_feat, module_kwargs=None
        )

        def replace_(llama_decoder_layer, layers_info) -> None:
            layers = layers_info
            for pair in layers:
                prev_op_name = pair["prev_op_name"]
                layers_lst = pair["layers"]
                scale_shape = layers_lst[0].in_features
                scale_mod = ScaleMod(scale_shape)
                layer_names = pair["layer_names"]
                for layer, layer_name in zip(layers_lst, layer_names):
                    awq_utils.set_module_name(llama_decoder_layer, layer_name, MulLinear(layer, scale_mod))
                _pre_op = awq_utils.get_module_by_name_suffix(llama_decoder_layer, prev_op_name)
                awq_utils.set_module_name(llama_decoder_layer, prev_op_name, DivLinear(_pre_op, scale_mod))

        def revert_replace(llama_decoder_layer, layer_info):
            scale_lst = []
            layers = layer_info
            for pair in layers[::-1]:
                prev_op_name = pair["prev_op_name"]
                layer_names = pair["layer_names"]
                div_linear = awq_utils.get_module_by_name_suffix(llama_decoder_layer, prev_op_name)
                awq_utils.set_module_name(llama_decoder_layer, prev_op_name, div_linear.linear)
                scale_lst.append((prev_op_name, layer_names, div_linear.scale_mod.get_best_scale()))
                for _layer_name in layer_names:
                    mul_layer = awq_utils.get_module_by_name_suffix(llama_decoder_layer, _layer_name)
                    awq_utils.set_module_name(llama_decoder_layer, _layer_name, mul_layer.linear)
            return llama_decoder_layer, scale_lst[::-1]

        def absorb_mul_(llama_decoder_layer, layers_info):
            llama_decoder_layer, scale_lst = revert_replace(llama_decoder_layer, layers_info)
            awq.quantize.scale.apply_scale(llama_decoder_layer, scale_lst)

        batch, seq_len, embed_dim = 2, 10, 4096
        position_ids = torch.arange(seq_len).unsqueeze(0)
        hidden_states = torch.randn(batch, seq_len, embed_dim)
        output_ref = llama_decoder_layer(hidden_states, position_ids=position_ids)

        replace_(llama_decoder_layer, layers_info=layers_info)
        output = llama_decoder_layer(hidden_states, position_ids=position_ids)
        assert_same(output_ref, output)

        absorb_mul_(llama_decoder_layer, layers_info=layers_info)
        output_fused_mul = llama_decoder_layer(hidden_states, position_ids=position_ids)
        assert_same(output_ref, output_fused_mul)

    def _expect_replace(self):
        """
                LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): MulLinear(
              (linear): Linear(in_features=4096, out_features=4096, bias=False)
              (scale_mod): ScaleMod(shape=torch.Size([4096]))
            )
            (k_proj): MulLinear(
              (linear): Linear(in_features=4096, out_features=4096, bias=False)
              (scale_mod): ScaleMod(shape=torch.Size([4096]))
            )
            (v_proj): DivLinear(
              (linear): MulLinear(
                (linear): Linear(in_features=4096, out_features=4096, bias=False)
                (scale_mod): ScaleMod(shape=torch.Size([4096]))
              )
              (scale_mod): ScaleMod(shape=torch.Size([4096]))
            )
            (o_proj): MulLinear(
              (linear): Linear(in_features=4096, out_features=4096, bias=False)
              (scale_mod): ScaleMod(shape=torch.Size([4096]))
            )
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): MulLinear(
              (linear): Linear(in_features=4096, out_features=11008, bias=False)
              (scale_mod): ScaleMod(shape=torch.Size([4096]))
            )
            (up_proj): DivLinear(
              (linear): MulLinear(
                (linear): Linear(in_features=4096, out_features=11008, bias=False)
                (scale_mod): ScaleMod(shape=torch.Size([4096]))
              )
              (scale_mod): ScaleMod(shape=torch.Size([11008]))
            )
            (down_proj): MulLinear(
              (linear): Linear(in_features=11008, out_features=4096, bias=False)
              (scale_mod): ScaleMod(shape=torch.Size([11008]))
            )
            (act_fn): SiLU()
          )
          (input_layernorm): DivLinear(
            (linear): LlamaRMSNorm()
            (scale_mod): ScaleMod(shape=torch.Size([4096]))
          )
          (post_attention_layernorm): DivLinear(
            (linear): LlamaRMSNorm()
            (scale_mod): ScaleMod(shape=torch.Size([4096]))
          )
        )
        """

    def test_replace_v2(self):
        import torch
        import transformers

        model_name = "/data5/yliu7/Llama-2-7b-chat-hf/"
        from transformers.models.llama import modeling_llama

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        llama_decoder_layer: modeling_llama.LlamaDecoderLayer = model.model.layers[0]
        model.eval()
        fake_input_feat = awq.models.llama.LlamaAWQForCausalLM.fake_input_feat()
        layers_info = awq.models.llama.LlamaAWQForCausalLM.get_layers_for_scaling(
            llama_decoder_layer, fake_input_feat, module_kwargs=None
        )

        module_pairs_info = _create_layer_info(layers_info)

        batch, seq_len, embed_dim = 2, 10, 4096
        position_ids = torch.arange(seq_len).unsqueeze(0)
        hidden_states = torch.randn(batch, seq_len, embed_dim)
        output_ref = llama_decoder_layer(hidden_states, position_ids=position_ids)

        replace_(llama_decoder_layer, module_pairs_info=module_pairs_info)
        output = llama_decoder_layer(hidden_states, position_ids=position_ids)
        print(llama_decoder_layer)
        assert_same(output_ref, output)
        if scale_mod_version == "1":
            assert (
                len(get_tranable_params(llama_decoder_layer)) == 4
            ), f"There should be 4 trainable parameters. Got {len(get_tranable_params(llama_decoder_layer))}"
        absorb_mul_(llama_decoder_layer, module_pairs_info=module_pairs_info)
        llama_decoder_layer = llama_decoder_layer.to(hidden_states.device)
        output_fused_mul = llama_decoder_layer(hidden_states, position_ids=position_ids)
        assert_same(output_ref, output_fused_mul)

    def test_flow(self):
        def quant_block_(block):
            trainable_param = block.get_tranable_params()
            for iter in range(niters):
                loss = ...
                loss.backward()

        # ==-----------------------------------------------------------------==
        model_name = "/data5/yliu7/Llama-2-7b-chat-hf/"
        from transformers.models.llama import modeling_llama

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
        llama_decoder_layer: modeling_llama.LlamaDecoderLayer = model.model.layers[0]
        model.eval()
        llama_decoder_layer.eval()
        model_type = "llama"

        batch, seq_len, embed_dim = 2, 10, 4096
        position_ids = torch.arange(seq_len).unsqueeze(0)
        hidden_states = torch.randn(batch, seq_len, embed_dim)
        output_ref = llama_decoder_layer(hidden_states, position_ids=position_ids)

        api_replace_(llama_decoder_layer, model_type)
        output = llama_decoder_layer(hidden_states, position_ids=position_ids)
        print(llama_decoder_layer)
        assert_same(output_ref, output)
        if scale_mod_version == "1":
            assert (
                len(get_tranable_params(llama_decoder_layer)) == 4
            ), f"There should be 4 trainable parameters. Got {len(get_tranable_params(llama_decoder_layer))}"
        api_absorb_mul_(llama_decoder_layer)
        llama_decoder_layer = llama_decoder_layer.to(hidden_states.device)
        output_fused_mul = llama_decoder_layer(hidden_states, position_ids=position_ids)
        assert_same(output_ref, output_fused_mul)
    
    @pytest.mark.skipif(scale_mod_version != "1", reason="Only works for ScaleModV1")
    def test_scale_mod(self):
        scale_mod = ScaleMod(10)
        scale_mod(torch.randn(1, 10))
        scale_mod.update_best_scale()
        assert torch.equal(scale_mod.scales, scale_mod._best_scales)
        scale_mod.scales = torch.nn.Parameter(scale_mod.scales + 1)
        assert not torch.equal(scale_mod.scales, scale_mod._best_scales)
        scale_mod.update_best_scale()
        assert torch.equal(scale_mod.scales, scale_mod._best_scales)


