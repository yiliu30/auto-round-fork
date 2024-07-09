# ==-------------------------------------------------------------------------==
seed = 0
import random
random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import numpy as np
np.random.seed(seed)
# ==-------------------------------------------------------------------------==

import copy
import shutil
import sys
import unittest

sys.path.insert(0, "..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # model_name = "facebook/opt-125m"
        model_name = "/models/TinyLlama-1.1B-Chat-v1.0/"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_remove_whole_block(self):
        layer_config={"model.decoder.layers.0.self_attn.k_proj":{"data_type":"float"},
                       "model.decoder.layers.0.self_attn.v_proj": {"data_type": "float"},
                       "model.decoder.layers.0.self_attn.q_proj": {"data_type": "float"},
                       "model.decoder.layers.0.self_attn.out_proj": {"data_type": "float"},
                       "model.decoder.layers.0.fc1": {"data_type": "float"},
                       "model.decoder.layers.0.fc2": {"data_type": "float"},
                       }
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config
        )
        autoround.quantize()

    def test_default_torch_woq_linear(self):
        import os
        os.environ["TORCH_WOQ"] = "1"
        bits, group_size, sym = 4, 32, False

        input_text = "Hi"

        input_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"]
        flaot_output = self.model(input_ids.to(self.model.device))
        # decode
        generate_ids = self.model.generate(input_ids, max_length=30)
        flaot_output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(f"!!!!!!!!!!!!!!! float_output: {flaot_output}")
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            enable_quanted_input=False
        )
        qmodel, _ = autoround.quantize()
        # input = torch.ones([1, 10], dtype=torch.long)
        qmodel = qmodel.to("cuda").to(torch.float32)
        output = qmodel(input_ids.to("cuda"))
        qmodel_generate_ids = qmodel.generate(input_ids.to("cuda"), max_length=30)
        qmodel_output = self.tokenizer.batch_decode(qmodel_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(f"!!!!!!!!!!!!!!! qmodel_output: {qmodel_output}")
        # qmodel_output: Hi! I'm interested in learning more about your services. Could you please provide me with more information about your team and their qualifications?
        # decode output to text
        import pdb; pdb.set_trace()
        # decode ids into text
        # if torch.cuda.is_available():
        #     autoround.save_quantized(output_dir="./saved", inplace=False)
        # autoround.save_quantized(output_dir="./saved", inplace=False, format="itrex")

    def test_default(self):
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        if torch.cuda.is_available():
            autoround.save_quantized(output_dir="./saved", inplace=False)
        autoround.save_quantized(output_dir="./saved", inplace=False, format="itrex")

    def test_sym(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_w4g1(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_w3g128(self):
        bits, group_size, sym = 3, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_w2g128(self):
        bits, group_size, sym = 2, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_disable_enable_quanted_input(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_quanted_input=False,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_disable_minmax_tuning(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_minmax_tuning=False,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_signround(self):
        bits, group_size, sym = 4, -1, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_minmax_tuning=False,
            enable_quanted_input=False,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_lm_head(self):
        bits, group_size, sym = 4, -1, False
        layer_config = {"lm_head": {"data_type": "int"}}
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_minmax_tuning=False,
            enable_quanted_input=False,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()
    def test_wa_quant(self):
        bits, group_size, sym, act_bits = 4, 128, False, 4
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            act_bits=4,
        )
        autoround.quantize()



if __name__ == "__main__":
    unittest.main()
