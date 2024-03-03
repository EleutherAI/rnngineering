import sys
sys.path.append("tuned-lens")
from tuned_lens.nn.lenses import Lens, LogitLens, TunedLens
from copy import deepcopy
from typing import List

import accelerate
import torch
from datasets import load_dataset, Dataset   
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from nnsight.models.Mamba import Mamba
from mamba_ssm.ops.triton.layernorm import rms_norm_fn

from tuned_lens.scripts.ingredients import Model

accelerator = accelerate.Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="mamba_lenses")


def kl_divergence(labels, outputs):
    return torch.sum(
        labels.exp() * (labels - outputs), dim=-1
    ).mean()

# func that does the last decode step after hook for either logitlens or tunedlens
# lens=None for logit lens, or give it a TunedLens and it'll do that 
def decode(output, layer=0, lens=None):
    hidden_states = output[0] + output[1]

    if lens is not None and layer != len(lens.layer_translators):
        hidden_states = lens.layer_translators[layer](hidden_states) + hidden_states

    norm_f = model.local_model.backbone.norm_f

    decoded = hidden_states.node.graph.add(
        target=rms_norm_fn,
        args=[hidden_states, norm_f.weight, norm_f.bias],
        kwargs={
            "eps": norm_f.eps,
            "residual": None,
            "prenorm": False,
            "residual_in_fp32": True,
        },
    )

    return model.lm_head(decoded)


class NN_sight_TunedLens(torch.nn.Module):
    def __init__(self, layers: List, d_model: int) -> None:
        super().__init__()

        translator = torch.nn.Linear(d_model, d_model, bias=True)
        translator.weight.data.zero_()
        translator.bias.data.zero_()

        self.layer_translators = torch.nn.ModuleList(
            [deepcopy(translator) for _ in range(len(layers) - 1)]
        )


def test(dataloader, model, nn_tuned_lens,tuned_lens, max_length):
    log_losses = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Test:", disable=not accelerator.is_local_main_process)):
            if batch_idx > 10:
                break
            with model.forward(validate=False, inference=True) as runner:
                with runner.invoke(
                    batch["text"], scan=False, truncation=True, max_length=max_length
                ) as invoker:
                    # get output probs by using "logit lens" on the last layer 
                    model_pred_probs = torch.nn.functional.log_softmax(
                        decode(model.backbone.layers[-1].output)[:, -1, :]
                    )

                    logit_kls = []
                    nn_tuned_kls = []
                    tuned_kls = []
                    for layer_idx, layer in enumerate(model.backbone.layers):
                        
                        logitlens_logprobs = torch.nn.functional.log_softmax(
                            decode(layer.output, layer=layer_idx), dim=-1 # (bsz, seq_len, vocab)
                        )[:, -1, :].save()
                        
                        logit_kls.append(
                            kl_divergence(model_pred_probs, logitlens_logprobs).save()
                        )
                        
                        nn_tunedlens_logprobs = torch.nn.functional.log_softmax(
                            decode(layer.output, layer=layer_idx, lens=nn_tuned_lens), dim=-1
                        )[:, -1, :].save() # (bsz, 50280)
                        
                        nn_tuned_kls.append(
                            kl_divergence(model_pred_probs, nn_tunedlens_logprobs).save()
                        )

                        tunedlens_logprobs = torch.nn.functional.log_softmax(
                            decode(layer.output, layer=layer_idx, lens=tuned_lens), dim=-1
                        )[:, -1, :].save()
                        
                        tuned_kls.append(
                            kl_divergence(model_pred_probs, tunedlens_logprobs).save()
                        )

            logit_kls = [t.value.cpu() for t in logit_kls]
            tuned_kls = [t.value.cpu() for t in tuned_kls]
            nn_tuned_kls = [t.value.cpu() for t in nn_tuned_kls]
            
            if log_losses is None:
                log_losses = {f"val_logit_kl_layer_{layer_idx}" : [kl] for layer_idx, kl in enumerate(logit_kls)}
                for layer_idx, kl in enumerate(tuned_kls):
                    log_losses[f"val_tuned_kl_layer_{layer_idx}"] = [kl]    
                for layer_idx, kl in enumerate(nn_tuned_kls):
                    log_losses[f"val_nn_tuned_kl_layer_{layer_idx}"] = [kl]
            else:
                for layer_idx, kl in enumerate(logit_kls):
                    log_losses[f"val_logit_kl_layer_{layer_idx}"].append(kl)
                for layer_idx, kl in enumerate(tuned_kls):
                    log_losses[f"val_tuned_kl_layer_{layer_idx}"].append(kl)
                for layer_idx, kl in enumerate(nn_tuned_kls):
                    log_losses[f"val_nn_tuned_kl_layer_{layer_idx}"].append(kl)

        log_losses = {key: torch.tensor(value).mean() for key, value in log_losses.items()}
        accelerator.log(log_losses, step=global_step)
    
    return log_losses

#CUDA_VISIBLE_DEVICES=7 python -m tuned_lens train --model.name RWKV/rwkv-4-169m-pile --data.name val.jsonl --per_gpu_batch_size=10 --output my_lens/rwkv/169m --checkpoint_dir my_lens/rwkv/169m --wandb 

#rwkv = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-4-169m-pile")
global_step = 0 

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id
#model = Mamba("state-spaces/mamba-1.4b", device="cuda:1",tokenizer=tokenizer, dispatch=True)
model = Mamba("state-spaces/mamba-130m", device="cuda:0",tokenizer=tokenizer, dispatch=True)

d_model_hidden_states = model.local_model.config.d_model

# Tuned lens is just ModuleList of linear layers
lens = NN_sight_TunedLens(model.backbone.layers, d_model_hidden_states).to("cuda:1")
# ckpt = '/share/u/jadenfk/wd/tunedlens_34/model.safetensors'
ckpt = 'mamba_lenses/tunedlens_34_mamba-130m.safetensors'
#ckpt = 'mamba_lenses/tunedlens_34_mamba-1.4b.safetensors'


#lens = lens.to(torch.bfloat16)
lens = accelerate.load_checkpoint_and_dispatch(lens, ckpt)

tuned_model = Model("state-spaces/mamba-130m",tokenizer="EleutherAI/gpt-neox-20b")
#tuned_model = Model("state-spaces/mamba-1.4b",tokenizer="EleutherAI/gpt-neox-20b")
model, tokenize = tuned_model.load("cuda:1")
tuned_lens = TunedLens.from_model_and_pretrained(model,"tuned-lens/my_lenses/mamba/130m")

model = Mamba("state-spaces/mamba-130m", device="cuda:0",tokenizer=tokenizer, dispatch=True)
#model = Mamba("state-spaces/mamba-1.4b", device="cuda:1",tokenizer=tokenizer, dispatch=True)

#dataset = load_dataset("JeanKaddour/minipile", data_dir="data")
dataset = Dataset.from_json("tuned-lens/val.jsonl")

val_dataloader = DataLoader(dataset, batch_size=30, shuffle=False)

(model.local_model, val_dataloader) = accelerator.prepare(model.local_model, val_dataloader)
 
logged_kls = test(val_dataloader, model, lens,tuned_lens, 25)

logit_kls = [logged_kls[f"val_logit_kl_layer_{i}"] for i, _ in enumerate(model.backbone.layers)]
tuned_kls = [logged_kls[f"val_tuned_kl_layer_{i}"] for i, _ in enumerate(model.backbone.layers)]
nn_tuned = [logged_kls[f"val_nn_tuned_kl_layer_{i}"] for i, _ in enumerate(model.backbone.layers)]
import matplotlib.pyplot as plt 
import numpy as np
x = [i for i, _ in enumerate(model.backbone.layers)]
plt.figure(figsize=(15,8))
plt.plot(x, np.array(logit_kls), label="Logit Lens")
plt.plot(x, np.array(tuned_kls), label="Tuned Lens")
plt.plot(x, np.array(nn_tuned), label="NN Tuned Lens")
plt.legend()
plt.xticks(x)
plt.xlabel("Layer")
plt.ylabel("KL Divergence (nats)")






