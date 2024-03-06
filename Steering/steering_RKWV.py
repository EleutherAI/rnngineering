import os
import numpy as np
import torch
from torch.nn import functional as F
import json
from tqdm import tqdm
from collections import OrderedDict

import matplotlib.pyplot as plt

import pandas as pd

from model_patches.hf_rwkv5 import Rwkv5ForCausalLM
from model_patches.configuration_rwkv5 import Rwkv5Config
from model_patches.rwkv_v5_utils import PIPELINE

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"  # '1' to compile CUDA kernel (10x faster)


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
            new_key = (
                new_key.replace("time_mix_k", "time_mix_key")
                .replace("time_mix_v", "time_mix_value")
                .replace("time_mix_g", "time_mix_gate")
                .replace("time_mix_r", "time_mix_receptance")
                .replace("ln0", "pre_ln")
                .replace("att", "attention")
                .replace("ffn", "feed_forward")
                .replace("emb", "embeddings")
            )
            new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


class ActivationAdder:
    def __init__(self):
        self.activations = []
        self.state = []

    def record_activations(self, layer, state=False):
        def hook(model, input, output):
            self.activations.append(output[0].detach()[:, -2, :].cpu())
            if state:
                self.state.append(
                    [
                        output[1][0].detach()[:, :, layer].cpu(),
                        output[1][1].detach()[:, :, :, :, layer].cpu(),
                        output[1][2].detach()[:, :, layer].cpu(),
                    ]
                )

        return hook

    def add_activations(self, act=None, mult=1, start_pos=-1, layer=0, state=False):
        if act is None:
            act = sum(self.activations) / len(self.activations)
            if state:
                st0 = 0
                st1 = 0
                st2 = 0
                for i in range(len(self.state)):
                    st0 = st0 + self.state[i][0]
                    st1 = st1 + self.state[i][1]
                    st2 = st2 + self.state[i][2]

                st0 = st0 / len(self.state)
                st1 = st1 / len(self.state)
                st2 = st2 / len(self.state)

        def hook(model, input, output):
            output[0][:, start_pos:, :] += mult * act.cuda(device)
            if state:
                output[1][0][:, :, layer] += mult * st0.cuda(device)
                output[1][1][:, :, :, :, layer] += mult * st1.cuda(device)
                output[1][2][:, :, layer] += mult * st2.cuda(device)

            return output

        return hook

    def save_activations(self, path):
        torch.save(self.activations, path)

    def save_state(self, path):
        torch.save(self.state, path)

    def load_activations(self, path):
        self.activations = torch.load(path)

    def load_state(self, path):
        self.state = torch.load(path)


MODEL_NAME = "/mnt/ssd-1/thomas/Hermes-RWKV-v5-7B.pth"
model = load_hf_rwkv5(MODEL_NAME)
tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")

device = "cuda:0"
model.cuda(device)
a_token_id = tokenizer.encode("A")
b_token_id = tokenizer.encode("B")
layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31]


dataset_path = "../datasets/generate/refusal/generate_dataset.json"
with open(dataset_path, "rb") as fd:
    dataset = json.load(fd)

actadds = {layer: ActivationAdder() for layer in layers}
counter = 0
for prompt in tqdm(dataset, desc="Generating steering vector"):
    counter += 1
    msg = prompt["question"]
    a_matches = prompt["answer_matching_behavior"] == "(A)"
    msg = "\x16user\n" + msg + "\x17\n\x16assistant\n"
    with torch.no_grad():
        positive_prompt = msg + "(A)" if a_matches else msg + "(B)"
        pos_activations = [
            model.rwkv.blocks[layer].register_forward_hook(
                actadds[layer].record_activations(layer)
            )
            for layer in layers
        ]
        model.forward(
            torch.tensor(tokenizer.encode(positive_prompt)).unsqueeze(0).cuda(device)
        )
        for h in pos_activations:
            h.remove()

        neg_activations = [
            model.rwkv.blocks[layer].register_forward_hook(
                actadds[layer].record_activations(layer)
            )
            for layer in layers
        ]
        negative_prompt = msg + "(B)" if a_matches else msg + "(A)"
        model.forward(
            torch.tensor(tokenizer.encode(negative_prompt)).unsqueeze(0).cuda(device)
        )
        for h in neg_activations:
            h.remove()

for layer in layers:
    actadds[layer].save_activations(f"rkwv-refusal_activations_{layer}.pt")
    actadds[layer].save_state(f"rkwv-refusal_state_{layer}.pt")


test_dataset_path = "../datasets/test/refusal/test_dataset_ab.json"
with open(test_dataset_path, "rb") as fd:
    test_dataset = json.load(fd)

for layer in layers:
    actadds[layer].load_activations(f"rkwv-refusal_activations_{layer}.pt")
    actadds[layer].load_state(f"rkwv-refusal_state_{layer}.pt")


results = []

for layer in layers:
    for pos_before in [-1]:
        for mult in [-0.5, -0.1, 0, 0.1, 0.5]:
            average_prob = 0
            count = 0

            module = model.rwkv.blocks[layer]
            matching_prob = 0
            not_matching_prob = 0
            correct = 0
            h = module.register_forward_hook(
                actadds[layer].add_activations(
                    mult=mult / np.abs(pos_before), start_pos=pos_before, layer=layer
                )
            )
            # for prompt in tqdm(test_dataset, desc="Processing prompts"):
            for prompt in test_dataset:
                count += 1
                msg = prompt["question"]
                a_matches = prompt["answer_matching_behavior"] == "(A)"
                msg = "\x16user\n" + msg + "\x17\n\x16assistant\n("

                with torch.no_grad():
                    out = model.forward(
                        torch.tensor(tokenizer.encode(msg)).unsqueeze(0).cuda(device)
                    )

                logits = out.logits[0, -1, :]
                probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
                a_prob = probs[a_token_id]
                b_prob = probs[b_token_id]
                max_prob = F.softmax(logits.float(), dim=-1).argmax().cpu().numpy()
                if max_prob == a_token_id and a_matches:
                    correct += 1
                if max_prob == b_token_id and not a_matches:
                    correct += 1

                behavior_prob = a_prob if a_matches else b_prob
                average_prob += behavior_prob / (a_prob + b_prob)
                matching_prob += behavior_prob
                not_matching_prob += a_prob if not a_matches else b_prob
            results.append(
                (
                    layer,
                    mult,
                    pos_before,
                    average_prob / count,
                    matching_prob / count,
                    not_matching_prob / count,
                    correct / count,
                )
            )
            h.remove()
            print(
                f"{layer}, {mult}, {average_prob / count}, {matching_prob / count},"
                f" {not_matching_prob / count}, {correct/count}"
            )


cleandata = []
colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "black",
    "pink",
    "brown",
]
for i, values in enumerate(results):
    cleandata.append(
        [values[0], values[1], values[2], values[3][0], values[4][0], values[5][0]]
    )

dataframe = pd.DataFrame(
    cleandata, columns=["Layer", "Mult", "Pos", "Average", "Matching", "Not Matching"]
)
dataframe.to_csv("rwkv-refusal.csv")

mults = dataframe["Mult"].unique()
for i, m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"] == -1]

    data_previous = data_previous[data_previous["Mult"] == m]
    plt.plot(
        data_previous["Layer"],
        data_previous["Matching"]
        / (data_previous["Matching"] + data_previous["Not Matching"]),
        color=colors[i],
        label=f"Mult: {m}",
    )
plt.legend()
plt.figure()
mults = dataframe["Mult"].unique()
for i, m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"] == -1]

    data_previous = data_previous[data_previous["Mult"] == m]
    plt.plot(
        data_previous["Layer"],
        data_previous["Matching"],
        color=colors[i],
        label=f"Mult: {m}",
    )
plt.legend()
plt.figure()
mults = dataframe["Mult"].unique()
for i, m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"] == -1]

    data_previous = data_previous[data_previous["Mult"] == m]
    plt.plot(
        data_previous["Layer"],
        data_previous["Not Matching"],
        color=colors[i],
        label=f"Mult: {m}",
    )
plt.legend()

plt.figure()
mults = dataframe["Mult"].unique()
for i, m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"] == -1]

    data_previous = data_previous[data_previous["Mult"] == m]
    plt.plot(
        data_previous["Layer"],
        data_previous["Not Matching"] + data_previous["Matching"],
        color=colors[i],
        label=f"Mult: {m}",
    )
plt.legend()

plt.figure()

mults = dataframe["Mult"].unique()
for i, m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"] == -1]

    data_previous = data_previous[data_previous["Mult"] == m]
    plt.plot(
        data_previous["Layer"],
        data_previous["Average"],
        color=colors[i],
        label=f"Mult: {m}",
    )
plt.legend()
