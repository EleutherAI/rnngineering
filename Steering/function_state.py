from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from transformers import AutoTokenizer,MambaForCausalLM,MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers import AutoModelForCausalLM

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import math
from utils import cached_property, get_layer_list

def convert_mamba_to_hf(model_name):
    original_config = load_config_hf(model_name)
    original_state_dict = load_state_dict_hf(model_name)
        
    mamba_config = MambaConfig()
    for key in original_config:
        if hasattr(mamba_config,key):
            setattr(mamba_config,key,original_config[key])
        if key == "d_model":
            setattr(mamba_config,"hidden_size",original_config[key])
            setattr(mamba_config,"intermediate_size",original_config[key]*2)
            setattr(mamba_config,"time_step_rank",math.ceil(original_config[key] / 16))
        if key == "n_layer":
            setattr(mamba_config,"num_hidden_layers",original_config[key])
        if key == "vocab_size":
            vocab_size = original_config[key]
            pad_vocab_size_multiple = original_config["pad_vocab_size_multiple"]

            if vocab_size % pad_vocab_size_multiple != 0:
                        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
            setattr(mamba_config,"vocab_size",vocab_size)
    original_state_dict["backbone.embeddings.weight"] = original_state_dict["backbone.embedding.weight"]
    original_state_dict.pop("backbone.embedding.weight")
    model = MambaForCausalLM(mamba_config)#AutoModelForCausalLM.from_pretrained("state-spaces/mamba-370m-hf")
    model.load_state_dict(original_state_dict)
    model.eval()
    return model


@dataclass
class ActivationAdder:
    activations: list = field(default_factory=list)

    mamba_style: bool = False

    @cached_property
    def steering_vector(self):
        # [2, num samples, num features]
        acts = torch.stack([torch.cat(self.positives), torch.cat(self.negatives)])

        # [num features]
        u = -acts.mean(1).diff(dim=0).squeeze(0)

        return u

    def record_activations(self):
        def hook(model, input, output):
            act = output if self.mamba_style else output[0]
            act = act.detach()[:, -1, :].cpu()
            activations.append(act)
        return hook

    def add_activations(self, mult=1, start_pos=-1):
        u = self.steering_vector

        def hook(model, input, output):
            hooked = output if self.mamba_style else output[0]
            hooked[:, start_pos:, :] += mult * u.to(hooked.device)
            return output

        return hook

@dataclass
class StateAdder:
    states: dict = field(default_factory=dict )
    
    @cached_property
    def steering_vector(self):
       #batch size, dim, num features
       
        conv_states = torch.stack([torch.stack(self.states["positive_conv"]), torch.stack(self.states["negative_conv"])])
        ssm_states = torch.stack([torch.stack(self.states["positive_ssm"]), torch.stack(self.states["negative_ssm"])])
        # [num features]
        return (-conv_states.mean(1).diff(dim=0),
               -ssm_states.mean(1).diff(dim=0))

    def record_state(self, state, positive: bool = True):
        if not self.states:
            self.states = {"positive_conv": [], "negative_conv": [], "positive_ssm": [], "negative_ssm": []}
        temp_state = deepcopy(state)
        conv_states = torch.cat(list(temp_state.conv_states.values())).detach().cpu()
        ssm_states = torch.cat(list(temp_state.ssm_states.values())).detach().cpu()
        
        if positive:
            self.states["positive_conv"].append(conv_states)
            self.states["positive_ssm"].append(ssm_states)
        else:
            self.states["negative_conv"].append(conv_states)
            self.states["negative_ssm"].append(ssm_states)


    def steer_state(self, state, mult=1, layer=None):
        conv, ssm = self.steering_vector
        
        conv_state = state.conv_states
        
        ssm_state = state.ssm_states
        
        conv_state[layer] += mult * conv[:,layer,:,:].to(conv_state[layer].device).type_as(conv_state[layer])
        ssm_state[layer] += mult * ssm[:,layer,:,:].to(ssm_state[layer].device).type_as(ssm_state[layer])
        for layer in conv_state:
            conv_state[layer] = conv_state[layer].type(torch.bfloat16)
            ssm_state[layer] = ssm_state[layer].type(torch.bfloat16)
        return state
            


TEMPLATES = {
    "hermes": "\x16user\n{}\x17\n\x16assistant\n(",
    "llama": "[INST] {} [/INST] (",
}


if __name__ == "__main__":
    dataset_root = Path(__file__).parent.parent / "dataset_files" 

    parser = ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str, default="capitalize.json"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--model", type=str, default="EleutherAI/Hermes-mamba-2.8b-slimpj-cDPO"
    )
    parser.add_argument(
        "--template", type=str, choices=("hermes", "llama"), default="hermes"
    )
    parser.add_argument(
        "--previous",type=int,choices=(-1,-3),default=-1
    )
    parser.add_argument(
        "--both", action="store_true"
    )
    args = parser.parse_args()


    template = TEMPLATES[args.template]
    print(f"Using template:\n{template}")

    is_mamba = "mamba" in args.model.lower()
    is_rwkv = "rwkv" in args.model.lower()
    cache_name = "state" if is_rwkv else "past_key_values"

    if is_mamba:
        
        # from mamba_ssm.utils.generation import InferenceParams
        if "hf" in args.model:
            model = AutoModelForCausalLM.from_pretrained(args.model, device=args.device)
        else :
            model = convert_mamba_to_hf(args.model).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    else:
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": args.device},
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)


    layer_list = get_layer_list(model)

    act_root = Path("functions") / args.model
    
    # Create all the activations

    print("Generating activations...")

    dataset_path = dataset_root / f"{args.dataset}.json"
    
    with open(dataset_path, "rb") as fd:
        dataset = json.load(fd)

    #split the dataset into train and test
    split_factor = 0.7
    train_size = int(len(dataset) * split_factor)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    if act_root.joinpath(f"{args.dataset}_state.pt").exists():
            print("Loading cached activations...")
            stateadder = StateAdder(**torch.load(act_root / f"{args.behavior}_state.pt"))
            if args.both:
                actadds = [
                    ActivationAdder(**torch.load(act_root / f"{args.behavior}_{layer}.pt"))
                    for layer in range(len(layer_list))
                ]
    else:
        stateadder = StateAdder()
        if args.both:
            actadds = [
                ActivationAdder(mamba_style="mamba" in args.model.lower())
                for _ in range(len(layer_list))
            ]
        for prompt in tqdm(train_dataset, desc="Generating steering vector"):
            msg = prompt["input"]
            msg = template.format(msg)
            msg = msg + prompt["output"]
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                if args.both:
                    activations = [
                        layer.register_forward_hook(adder.record_activations())
                        for adder, layer in zip(actadds, layer_list)
                    ]
                stateadder.record_state(
                    model(
                    torch.tensor(tokenizer.encode(msg))
                    .unsqueeze(0)
                    .to(args.device),use_cache=True,cache_params=None).cache_params,
                    positive=True)
                if args.both:
                    for h in activations:
                        h.remove()

        
        path = act_root / f"{args.dataset}_state.pt"
        print(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(asdict(stateadder), path)
        if args.both:
            for layer in range(len(layer_list)):
                path = act_root / f"{args.dataset}_{layer}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)

                torch.save(asdict(actadds[layer]), path)
    # Rows in a dataframe
    records: list[dict] = []

    for prompt in tqdm(test_dataset):
        msg = prompt["question"]
        a_matches = prompt["answer_matching_behavior"] == "(A)"
        msg = template.format(msg)

        inputs = torch.tensor(tokenizer.encode(msg)).unsqueeze(0).to(args.device)

        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            # We're only applying the steering vector to the last token
            prefix, suffix = inputs[:, :args.previous], inputs[:, args.previous:]

            cache = model(prefix, use_cache=True).cache_params

            for i, layer in enumerate(get_layer_list(model)):
                for mult in [-3,-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5,3.0]:
                    
                    if args.both:
                         h = layer.register_forward_hook(
                        actadds[i].add_activations(mult=mult, start_pos=args.previous)
                    )
                    if abs(args.previous) == 1:
                        new_cache=stateadder.steer_state(deepcopy(cache), mult, i)
                        logits = model(suffix, cache_params=new_cache).logits[0, -1]
                    else:
                        for j in range(abs(args.previous)-1):
                            new_cache=stateadder.steer_state(deepcopy(cache), mult, i)
                            cache = model(suffix[:,j].unsqueeze(0), cache_params=new_cache).cache_params
                        new_cache=stateadder.steer_state(deepcopy(cache), mult, i)
                        logits = model(suffix[:,-1].unsqueeze(0), cache_params=new_cache).logits[0, -1]
                    if args.both:
                        h.remove()
                    
                    probs = logits.softmax(-1)
                    a_prob = probs[a_token_id].item()
                    b_prob = probs[b_token_id].item()

                    matching_prob = (a_prob if a_matches else b_prob) / (
                        a_prob + b_prob
                    )
                    nonsense_prob = 1 - (a_prob + b_prob)

                    records.append(
                        {
                            "layer": i,
                            "multiplier": mult,
                            "matching": matching_prob,
                            "nonsense": nonsense_prob,
                        }
                    )

    df = pd.DataFrame.from_records(records)
    stats = df.groupby(["layer", "multiplier"]).mean()
    name = "caa_both" if args.both else "caa_state"
    if args.system:
        args.format = "system"
    path = (
        Path("results")
        / args.model
        / ("lda" if args.lda else name)
        / f"{args.behavior}"
        / f"{abs(args.previous)}"
        / f"{args.format}.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path)