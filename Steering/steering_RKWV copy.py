import os, copy, types, gc, sys, re
import numpy as np
import torch
from torch.nn import functional as F
import json
from tqdm import tqdm
from collections import OrderedDict

from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig,PretrainedConfig,PreTrainedModel

from hf_rwkv5 import Rwkv5ForCausalLM
from configuration_rwkv5 import Rwkv5Config
from rwkv_v5_utils import PIPELINE

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries


def load_hf_rwkv5(model_path):
    config = Rwkv5Config(num_hidden_layers=32, hidden_size=4096)
    torch.set_default_dtype(torch.bfloat16)
    model = Rwkv5ForCausalLM(config)
    state_dict = torch.load(model_path)

    new_state_dict = OrderedDict()
    for key in state_dict:
        if key == "head.weight":
            new_state_dict[key] = state_dict[key]
        else:   
            new_key = f"rwkv.{key}"
            new_key = (new_key
                        .replace("time_mix_k", "time_mix_key")
                        .replace("time_mix_v", "time_mix_value")
                        .replace("time_mix_g", "time_mix_gate")
                        .replace("time_mix_r", "time_mix_receptance")
                        .replace("ln0", "pre_ln")
                        .replace("att", "attention")
                        .replace("ffn", "feed_forward")
                        .replace("emb", "embeddings"))
            new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()
    return model



class ActivationAdder:
    def __init__(self):
        self.activations = []

    def record_activations(self, mult = 1):
        def hook(model, input, output):
            self.activations.append(mult * output[0].detach()[:, -1, :])
        return hook

    def add_activations(self, act = None, mult = 1):
        if act is None:
            act = sum(self.activations) / len(self.activations)
        def hook(model, input, output):
            output[0][:, -1, :] += mult * act
            return output[0], output[1]
        return hook

name= "state-spaces/mamba-2.8b-slimpj"
config_data = load_config_hf(name)
m_config = MambaConfig(**config_data)
model = MambaLMHeadModel(m_config).to('cuda:0')
state_dict = load_state_dict_hf(name)
model.load_state_dict(state_dict)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
a_token_id = tokenizer.encode('A')
b_token_id = tokenizer.encode('B')
layers = [32]

dataset_path = "../CAA/datasets/generate/sycophancy/generate_dataset.json"
with open(dataset_path, 'rb') as fd:
    dataset = json.load(fd)

actadds = {layer: ActivationAdder() for layer in layers}
for prompt in tqdm(dataset, desc="Generating steering vector"):
    msg = prompt['question']
    a_matches = (prompt['answer_matching_behavior'] == '(A)')
    with torch.no_grad():

        positive_prompt = msg + "(A)" if a_matches else msg + "(B)"
        pos_activations = [model.backbone.layers[layer].register_forward_hook(actadds[layer].record_activations())
                for layer in layers]
        model.forward(torch.tensor(tokenizer.encode(positive_prompt)).unsqueeze(0).cuda(0))
        for h in pos_activations: h.remove()

        neg_activations = [model.backbone.layers[layer].register_forward_hook(actadds[layer].record_activations())
                for layer in layers]
        negative_prompt = msg + "(B)" if a_matches else msg + "(A)"
        model.forward(torch.tensor(tokenizer.encode(negative_prompt)).unsqueeze(0).cuda(0))
        for h in neg_activations: h.remove()

test_dataset_path = "../CAA/datasets/test/sycophancy/test_dataset_ab.json"
with open(test_dataset_path, 'rb') as fd:
    test_dataset = json.load(fd)

for layer in layers:
    for mult in [-3,-2,-1, 0, 1,2,3]:
        average_prob = 0
        count = 0

        module = model.backbone.layers[layer]
        matching_prob = 0
        not_matching_prob = 0
        h = module.register_forward_hook(actadds[layer].add_activations(mult=mult))
        #for prompt in tqdm(test_dataset, desc="Processing prompts"):
        for prompt in test_dataset:
            count += 1
            msg = prompt['question']
            a_matches = (prompt['answer_matching_behavior'] == '(A)')
            msg = msg + "Answer: ("
            with torch.no_grad():
                out = model.forward(torch.tensor(tokenizer.encode(msg)).unsqueeze(0).cuda(0))

            logits = out.logits[0, -1, :]
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            a_prob = probs[a_token_id]
            b_prob = probs[b_token_id]

            behavior_prob = a_prob if a_matches else b_prob
            average_prob += behavior_prob / (a_prob + b_prob)
            matching_prob += behavior_prob
            not_matching_prob += a_prob if not a_matches else b_prob
        h.remove()
        print(f"{layer}, {mult}, {average_prob / count}, {matching_prob / count}, {not_matching_prob / count}")
