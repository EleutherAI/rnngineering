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
    positives: list = field(default_factory=list)
    negatives: list = field(default_factory=list)

    lda: bool = False
    mamba_style: bool = False

    @cached_property
    def steering_vector(self):
        # [2, num samples, num features]
        acts = torch.stack([torch.cat(self.positives), torch.cat(self.negatives)])

        # [num features]
        u = -acts.mean(1).diff(dim=0).squeeze(0)
        if self.lda:
            # Compute precision matrix
            prec = torch.linalg.pinv(acts.flatten(0, 1).T.cov().float()).type_as(u)
            u = prec @ u

        return u

    def record_activations(self, positive: bool = True):
        def hook(model, input, output):
            act = output if self.mamba_style else output[0]
            act = act.detach()[:, -1, :].cpu()
            if positive:
                self.positives.append(act)
            else:
                self.negatives.append(act)

        return hook

    def add_activations(self, mult=1, start_pos=-1):
        u = self.steering_vector

        def hook(model, input, output):
            hooked = output if self.mamba_style else output[0]
            hooked[:, start_pos:, :] += mult * u.to(hooked.device)
            return output

        return hook


def change_format(msg,format):
    if format == "a-b":
        #a_token_id = tokenizer.encode("(A")[-1]
        #b_token_id = tokenizer.encode("(B")[-1]
        a_token_id = tokenizer.encode("(A")[-1]
        b_token_id = tokenizer.encode("(B")[-1]
    elif format == "1-2":
        a_token_id = tokenizer.encode("(1")[-1]
        b_token_id = tokenizer.encode("(2")[-1]
        msg = msg.replace("(A)","(1)").replace("(B)","(2)")
    #print(a_token_id,b_token_id)
    
    return msg,a_token_id,b_token_id

TEMPLATES = {
    "hermes": "\x16user\n{}\x17\n\x16assistant\n(",
    "llama": "[INST] {} [/INST] (",
}


if __name__ == "__main__":
    dataset_root = Path(__file__).parent.parent / "datasets" / "generate"

    parser = ArgumentParser()
    parser.add_argument(
        "behavior",
        type=str,
        choices=[folder.stem for folder in dataset_root.iterdir()],
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--lda", action="store_true")
    parser.add_argument(
        "--model", type=str, default="EleutherAI/Hermes-btlm-3b-8k"
    )
    parser.add_argument(
        "--template", type=str, choices=("hermes", "llama"), default="hermes"
    )
    parser.add_argument(
        "--format", type=str, choices=("a-b", "1-2"), default="a-b"
    )
    parser.add_argument("--previous", type=int, default=-1)

    args = parser.parse_args()

    template = TEMPLATES[args.template]
    print(f"Using template:\n{template}")

    is_mamba = "mamba" in args.model.lower()
    is_rwkv = "rwkv" in args.model.lower()
    cache_name = "state" if is_rwkv else "past_key_values"

    if is_mamba:
        if "hf" in args.model:
            model = AutoModelForCausalLM.from_pretrained(args.model, device=args.device)
        else :
            model = convert_mamba_to_hf(args.model).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": args.device},
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir="/mnt/ssd-1/hf_cache/hub"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,cache_dir="/mnt/ssd-1/hf_cache/hub")

    
    layer_list = get_layer_list(model)

    act_root = Path("activations") / args.model
    if act_root.joinpath(f"{args.behavior}_0.pt").exists():
         print("Loading cached activations...")
         actadds = [
             ActivationAdder(**torch.load(act_root / f"{args.behavior}_{layer}.pt"))
             for layer in range(len(layer_list))
         ]
    # Create all the activations
    else:
        print("Generating activations...")

        dataset_path = dataset_root / args.behavior / "generate_dataset.json"
        with open(dataset_path, "rb") as fd:
            dataset = json.load(fd)

        actadds = [
            ActivationAdder(mamba_style="mamba" in args.model.lower())
            for _ in range(len(layer_list))
        ]
        for prompt in tqdm(dataset, desc="Generating steering vector"):
            msg = prompt["question"]
            a_matches = prompt["answer_matching_behavior"] == "(A)"
            msg = template.format(msg)

            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                positive_prompt = msg + "A" if a_matches else msg + "B"
                pos_activations = [
                    layer.register_forward_hook(adder.record_activations(positive=True))
                    for adder, layer in zip(actadds, layer_list)
                ]
                model(
                    torch.tensor(tokenizer.encode(positive_prompt))
                    .unsqueeze(0)
                    .to(args.device)
                )
                for h in pos_activations:
                    h.remove()

                neg_activations = [
                    layer.register_forward_hook(
                        adder.record_activations(positive=False)
                    )
                    for adder, layer in zip(actadds, layer_list)
                ]
                negative_prompt = msg + "B" if a_matches else msg + "A"
                model(
                    torch.tensor(tokenizer.encode(negative_prompt))
                    .unsqueeze(0)
                    .to(args.device)
                )
                for h in neg_activations:
                    h.remove()

        for layer in range(len(layer_list)):
            path = act_root / f"{args.behavior}_{layer}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(asdict(actadds[layer]), path)

    test_dataset_path = (
        dataset_root.parent / f"test/{args.behavior}/test_dataset_ab.json"
    )
    with open(test_dataset_path, "rb") as fd:
        test_dataset = json.load(fd)

    # Slightly hacky
    for adder in actadds:
        adder.lda = args.lda

    # Rows in a dataframe
    records: list[dict] = []
    start_pos = args.previous
    for prompt in tqdm(test_dataset):
        msg = prompt["question"]
        a_matches = prompt["answer_matching_behavior"] == "(A)"
        msg = template.format(msg)
        msg,a_token_id,b_token_id = change_format(msg,args.format)
        inputs = torch.tensor(tokenizer.encode(msg)).unsqueeze(0).to(args.device)

        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            # We're only applying the steering vector to the last token
            prefix, suffix = inputs[:, :start_pos], inputs[:, start_pos:]

            if is_mamba:
                from mamba_ssm.utils.generation import InferenceParams

                # See example in https://github.com/state-spaces/mamba/issues/187
                #params = InferenceParams(max_seqlen=2048, max_batch_size=1)
                #model(prefix, inference_params=params)  # In-place update

                #params.seqlen_offset = prefix.shape[-1]
                #_kwargs = dict(inference_params=params)
            else:
                _kwargs = {cache_name: model(prefix, use_cache=True)[cache_name]}

            for i, layer in enumerate(get_layer_list(model)):
                for mult in [-3,-1.5, -0.5,  0, 0.5, 1.5, 3]:
                    # Add the appropriate forward hook
                    
                    h = layer.register_forward_hook(
                        actadds[i].add_activations(mult=mult, start_pos=start_pos)
                    )
                    # RNNs in-place modify the state which we don't want here
                    #kwargs = deepcopy(_kwargs) if is_mamba or is_rwkv else _kwargs

                    if is_mamba:
                        logits = model(inputs).logits[0, -1]
                    else:
                        kwargs = deepcopy(_kwargs) if is_rwkv else _kwargs
                        logits = model(suffix, **kwargs).logits[0, -1]

                    # Make sure to remove it!
                    h.remove()

                    probs = logits.softmax(-1)
                    a_prob = probs[a_token_id].sum().item()
                    b_prob = probs[b_token_id].sum().item()

                    matching_prob = (a_prob if a_matches else b_prob) / (
                        a_prob + b_prob
                    )
                    nonsense_prob = 1 - (a_prob + b_prob)
                    if nonsense_prob>0.1:
                        # print(nonsense_prob)
                        # print(a_token_id,b_token_id)
                    
                        # print(tokenizer.decode(probs.sort(descending=True).indices[:3].tolist()),probs.sort(descending=True).indices[:3],probs.sort(descending=True)[:3])
                    
                    records.append(
                        {
                            "layer": i,
                            "multiplier": mult,
                            "matching": matching_prob,
                            "nonsense": nonsense_prob,
                        }
                    )
        #print(records)
    df = pd.DataFrame.from_records(records)
    stats = df.groupby(["layer", "multiplier"]).mean()
    print(stats)
    path = (
        Path("results")
        / args.model
        / ("lda" if args.lda else "caa")
        / f"{args.behavior}"
        / f"{abs(args.previous)}"
        / f"{args.format}.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path)
