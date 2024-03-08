from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from .utils import cached_property, get_layer_list


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
            prec = torch.linalg.pinv(acts.flatten(0, 1).T.cov())
            u = prec @ u

        # Normalize
        # u /= u.norm()

        return u

    def record_activations(self, positive: bool = True):
        def hook(model, input, output):
            act = output[0] + output[1] if self.mamba_style else output[0]
            act = act.detach()[:, -2, :].cpu()
            if positive:
                self.positives.append(act)
            else:
                self.negatives.append(act)

        return hook

    def add_activations(self, mult=1, start_pos=-1):
        u = self.steering_vector

        def hook(model, input, output):
            output[0][:, start_pos, :] += mult * u.to(output[0].device)
            return output

        return hook


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
        "--model", type=str, default="EleutherAI/Hermes-mamba-2.8b-slimpj-cDPO"
    )
    parser.add_argument(
        "--template", type=str, choices=("hermes", "llama"), default="hermes"
    )
    args = parser.parse_args()

    template = TEMPLATES[args.template]
    print(f"Using template:\n{template}")

    # For transformers
    cache_name = "state" if "rwkv" in args.model.lower() else "past_key_values"

    # Kind of a hack, maybe fix later
    if "mamba" in args.model.lower():
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        # from mamba_ssm.utils.generation import InferenceParams
        from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

        config_data = load_config_hf(args.model)
        m_config = MambaConfig(**config_data)
        model = MambaLMHeadModel(m_config).to(args.device)
        state_dict = load_state_dict_hf(args.model)
        model.load_state_dict(state_dict)

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map={"": args.device}, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    a_token_id = tokenizer.encode("A")[-1]
    b_token_id = tokenizer.encode("B")[-1]

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
                positive_prompt = msg + "A)" if a_matches else msg + "B)"
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
                negative_prompt = msg + "B)" if a_matches else msg + "A)"
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

    # Rows in a dataframe
    records: list[dict] = []

    for prompt in tqdm(test_dataset):
        msg = prompt["question"]
        a_matches = prompt["answer_matching_behavior"] == "(A)"
        msg = template.format(msg)

        inputs = torch.tensor(tokenizer.encode(msg)).unsqueeze(0).to(args.device)

        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            # We're only applying the steering vector to the last token
            prefix, suffix = inputs[:, :-1], inputs[:, -1:]

            # Pre-compute the KV cache or state for the prefix
            kwargs = {cache_name: model(prefix, use_cache=True)[cache_name]}

            for i, layer in enumerate(get_layer_list(model)):
                for mult in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]:
                    # Add the appropriate forward hook
                    h = layer.register_forward_hook(
                        actadds[i].add_activations(mult=mult, start_pos=-1)
                    )
                    logits = model(suffix, **kwargs).logits[0, -1, :]

                    # Make sure to remove it!
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

    path = Path("results") / args.model / f"{args.behavior}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path)
