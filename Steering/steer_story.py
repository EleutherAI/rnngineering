from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import cached_property, get_layer_list
import torch.nn.functional as F


@dataclass
class StateAdder:
    positive_state_attn_xs: list = field(default_factory=list)
    positive_state_attn_kvs: list = field(default_factory=list)
    positive_state_ffn_xs: list = field(default_factory=list)
    negative_state_attn_xs: list = field(default_factory=list)
    negative_state_attn_kvs: list = field(default_factory=list)
    negative_state_ffn_xs: list = field(default_factory=list)

    @cached_property
    def steering_vector(self):
       # [2, num samples, num features]
       state_attn_xs = torch.stack([torch.cat(self.positive_state_attn_xs), torch.cat(self.negative_state_attn_xs)])
       state_attn_kvs = torch.stack([torch.cat(self.positive_state_attn_kvs), torch.cat(self.negative_state_attn_kvs)])
       state_ffn_xs = torch.stack([torch.cat(self.positive_state_ffn_xs), torch.cat(self.negative_state_ffn_xs)])

        # [num features]
       return (-state_attn_xs.mean(1).diff(dim=0),
               -state_attn_kvs.mean(1).diff(dim=0),
               -state_ffn_xs.mean(1).diff(dim=0))

    def record_state(self, state, positive: bool = True):
        state_attn_x = state[0].detach().cpu()
        state_attn_kv = state[1].detach().cpu()
        state_ffn_x = state[2].detach().cpu()

        if positive:
            self.positive_state_attn_xs.append(state_attn_x)
            self.positive_state_attn_kvs.append(state_attn_kv)
            self.positive_state_ffn_xs.append(state_ffn_x)
        else:
            self.negative_state_attn_xs.append(state_attn_x)
            self.negative_state_attn_kvs.append(state_attn_kv)
            self.negative_state_ffn_xs.append(state_ffn_x)


    def steer_state(self, state, mult=1, layer=None):
        a, b, c = self.steering_vector
        
        state_attn_x = state[0]
        state_attn_kv = state[1]
        state_ffn_x = state[2]
        
        if layer is None:
            return [
                state_attn_x + mult * a.to(state_attn_x.device()),
                state_attn_kv + mult * b.to(state_attn_kv.device()),
                state_ffn_x + mult * c.to(state_ffn_x.device())
            ]
        else:
            state_attn_x[:, :, layer] += mult * a[:, :, layer].to(state_attn_x.device)
            state_attn_kv[:, :, :, :, layer] += mult * b[:, :, :, :, layer].to(state_attn_kv.device)
            state_ffn_x[:, :, layer] += mult * c[:, :, layer].to(state_ffn_x.device)
            return [
                state_attn_x,
                state_attn_kv,
                state_ffn_x
            ]
            


TEMPLATES = {
    "hermes": "\x16user\n{}\x17\n\x16assistant\n",
    "llama": "[INST] {} [/INST] ",
}


if __name__ == "__main__":
    dataset_root = Path(__file__).parent.parent / "datasets" / "generate"

    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lda", action="store_true")
    parser.add_argument(
        "--model", type=str, default="EleutherAI/Hermes-mamba-2.8b-slimpj-cDPO"
    )
    parser.add_argument(
        "--template", type=str, choices=("hermes", "llama"), default="hermes"
    )
    args = parser.parse_args()

    template = TEMPLATES[args.template]
    print(f"Using template:\n{template}")

    is_rwkv = "rwkv" in args.model.lower()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": args.device},
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    layer_list = get_layer_list(model)

    print("Generating activations...")

    stateadder = StateAdder()
    positive_msg = "Write about hate."
    negative_msg = "Write about love."

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        positive_prompt = template.format(positive_msg)
        stateadder.record_state(
            model(
            torch.tensor(tokenizer.encode(positive_prompt))
            .unsqueeze(0)
            .to(args.device)).state,
            positive=True)

        negative_prompt = template.format(negative_msg)
        stateadder.record_state(
            model(
            torch.tensor(tokenizer.encode(negative_prompt))
            .unsqueeze(0)
            .to(args.device)).state,
            positive=False)


    msg = "Write a story about dogs."
    prompt = template.format(msg)
    mult = 1.

    inputs = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(args.device)

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        end_token = tokenizer.encode('\x17')[-1]
        out = model(inputs[:, :-1], state=None)
        state = out.state

        #RNNs in-place modify the state which we don't want here
        for i, layer in enumerate(get_layer_list(model)):
            state = stateadder.steer_state(state, mult, i)

        out = model(inputs[:, -1:], state=state)
        state = out.state
        logits = out.logits[0, -1]

        token = torch.multinomial(F.softmax(logits, dim=0), 1)
        print(tokenizer.decode(token), end='')


        for i in range(512):
            out = model(token.unsqueeze(0).to(args.device), state=state)
            logits = out.logits[0, -1]
            state = out.state
            token = torch.multinomial(F.softmax(logits, dim=0), 1)
            if token[-1] == end_token:
                break
            print(tokenizer.decode(token), end='', flush=True)
        print()