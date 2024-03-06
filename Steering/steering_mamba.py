from argparse import ArgumentParser
import json

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer


class ActivationAdder:
    def __init__(self):
        self.activations = []

    def record_activations(self):
        def hook(model, input, output):
            act = output[0] + output[1]
            self.activations.append(act.detach()[:, -2:, :].cpu())

        return hook

    def add_activations(self, act=None, mult=1, start_pos=-1):
        if act is None:
            act = sum(self.activations) / len(self.activations)

        def hook(model, input, output):
            output[0][:, start_pos:, :] += mult * act.cuda(device)
            return output

        return hook

    def save_activations(self, path):
        torch.save(self.activations, path)

    def load_activations(self, path):
        self.activations = torch.load(path)


def get_layer_list(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Get "the" list of layers from a model.

    This is operationalized as the unique `nn.ModuleList` that contains
    more than half of all the parameters in the model, if it exists.

    Args:
        model: The model to search.

    Returns:
        The nn.ModuleList.

    Raises:
        ValueError: If no such list exists.
    """
    total_params = sum(p.numel() for p in model.parameters())
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList):
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > total_params / 2:
                return module

    raise ValueError(
        "Could not find suitable `ModuleList`; is this an encoder-decoder model?"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument(
        "--model", type=str, default="EleutherAI/Hermes-mamba-2.8b-slimpj-cDPO"
    )
    args = parser.parse_args()

    # Kind of a hack, maybe fix later
    if "mamba" in args.model.lower():
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

        config_data = load_config_hf(args.model)
        m_config = MambaConfig(**config_data)
        model = MambaLMHeadModel(m_config).to(args.device)
        state_dict = load_state_dict_hf(args.model)
        model.load_state_dict(state_dict)

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    elif "rwkv" in args.model.lower():
        from model_patches.rwkv_v5_utils import load_hf_rwkv5, PIPELINE

        model = load_hf_rwkv5(args.model)
        tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = "cuda:2"

    a_token_id = tokenizer.encode("A")
    b_token_id = tokenizer.encode("B")

    layer_list = get_layer_list(model)
    layer_indices = list(range(0, 63, 2)) + [63]

    dataset_path = "../datasets/generate/survival-instinct/generate_dataset.json"
    with open(dataset_path, "rb") as fd:
        dataset = json.load(fd)

    actadds = {layer: ActivationAdder() for layer in layer_indices}
    for prompt in tqdm(dataset, desc="Generating steering vector"):
        msg = prompt["question"]
        a_matches = prompt["answer_matching_behavior"] == "(A)"
        msg = "\x16user\n" + msg + "\x17\n\x16assistant\n"

        with torch.no_grad():
            positive_prompt = msg + "(A)" if a_matches else msg + "(B)"
            pos_activations = [
                model.backbone.layers[layer].register_forward_hook(
                    actadds[layer].record_activations()
                )
                for layer in layer_indices
            ]
            model.forward(
                torch.tensor(tokenizer.encode(positive_prompt))
                .unsqueeze(0)
                .cuda(device)
            )
            for h in pos_activations:
                h.remove()

            neg_activations = [
                model.backbone.layers[layer].register_forward_hook(
                    actadds[layer].record_activations()
                )
                for layer in layer_indices
            ]
            negative_prompt = msg + "(B)" if a_matches else msg + "(A)"
            model.forward(
                torch.tensor(tokenizer.encode(negative_prompt))
                .unsqueeze(0)
                .cuda(device)
            )
            for h in neg_activations:
                h.remove()

    for layer in layer_indices:
        actadds[layer].save_activations(f"slimpj-dpo-survival-instinct_{layer}.pt")

    test_dataset_path = "../datasets/test/survival-instinct/test_dataset_ab.json"
    with open(test_dataset_path, "rb") as fd:
        test_dataset = json.load(fd)

    for layer in layer_indices:
        actadds[layer].load_activations(f"slimpj-dpo-survival-instinct_{layer}.pt")
    results = []
    for layer in layer_indices:
        for mult in [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]:
            average_prob = 0
            count = 0

            module = model.backbone.layers[layer]
            matching_prob = 0
            not_matching_prob = 0
            correct = 0
            h = module.register_forward_hook(
                actadds[layer].add_activations(mult=mult, start_pos=-1)
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
                    -1,
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
            [
                values[0],
                values[1],
                values[2],
                values[3][0],
                values[4][0],
                values[5][0],
                values[6],
            ]
        )

    dataframe = pd.DataFrame(
        cleandata,
        columns=[
            "Layer",
            "Mult",
            "Pos",
            "Average",
            "Matching",
            "Not Matching",
            "Correct",
        ],
    )
    dataframe.to_csv("slimpj-dpo-survival-instincts.csv")

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
        if abs(m) > 0.5:
            continue

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
        if abs(m) > 0.5:
            continue
        data_previous = data_previous[data_previous["Mult"] == m]
        plt.plot(
            data_previous["Layer"],
            data_previous["Average"],
            color=colors[i],
            label=f"Mult: {m}",
        )
    plt.legend()

    mults = dataframe["Mult"].unique()
    for i, m in enumerate(mults):
        data_previous = dataframe[dataframe["Pos"] == -1]
        if abs(m) > 0.5:
            continue
        data_previous = data_previous[data_previous["Mult"] == m]
        plt.plot(
            data_previous["Layer"],
            data_previous["Correct"],
            color=colors[i],
            label=f"Mult: {m}",
        )
    plt.legend()
