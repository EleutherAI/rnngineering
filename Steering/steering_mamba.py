from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import torch


class ActivationAdder:
    def __init__(self):
        self.activations = []

    def record_activations(self):
        def hook(model, input, output):
            act = output[0]+output[1]
            self.activations.append(act.detach()[:, -2:, :].cpu())
        return hook

    def add_activations(self, act = None, mult = 1,start_pos=-1):
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


device="cuda:2"
name= "EleutherAI/Hermes-mamba-2.8b-slimpj-cDPO"
config_data = load_config_hf(name)
m_config = MambaConfig(**config_data)
model = MambaLMHeadModel(m_config).to(device)
state_dict = load_state_dict_hf(name)
model.load_state_dict(state_dict)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
a_token_id = tokenizer.encode('A')
b_token_id = tokenizer.encode('B')
layers = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,63]

dataset_path = "../datasets/generate/survival-instinct/generate_dataset.json"
with open(dataset_path, 'rb') as fd:
    dataset = json.load(fd)

actadds = {layer: ActivationAdder() for layer in layers}
for prompt in tqdm(dataset, desc="Generating steering vector"):
    msg = prompt['question']
    a_matches = (prompt['answer_matching_behavior'] == '(A)')
    msg = "\x16user\n" + msg + "\x17\n\x16assistant\n"

    with torch.no_grad():
        positive_prompt = msg + "(A)" if a_matches else msg + "(B)"
        pos_activations = [model.backbone.layers[layer].register_forward_hook(actadds[layer].record_activations())
                for layer in layers]
        model.forward(torch.tensor(tokenizer.encode(positive_prompt)).unsqueeze(0).cuda(device))
        for h in pos_activations: h.remove()

        neg_activations = [model.backbone.layers[layer].register_forward_hook(actadds[layer].record_activations())
                for layer in layers]
        negative_prompt = msg + "(B)" if a_matches else msg + "(A)"
        model.forward(torch.tensor(tokenizer.encode(negative_prompt)).unsqueeze(0).cuda(device))
        for h in neg_activations: h.remove()

for layer in layers:
    actadds[layer].save_activations(f"slimpj-dpo-survival-instinct_{layer}.pt")

test_dataset_path = "../datasets/test/survival-instinct/test_dataset_ab.json"
with open(test_dataset_path, 'rb') as fd:
    test_dataset = json.load(fd)

for layer in layers:
    actadds[layer].load_activations(f"slimpj-dpo-survival-instinct_{layer}.pt")
results = []
for layer in layers:
    for mult in [-1,-0.5,-0.1,0,0.1,0.5,1]:
        average_prob = 0
        count = 0

        module = model.backbone.layers[layer]
        matching_prob = 0
        not_matching_prob = 0
        correct=0
        h = module.register_forward_hook(actadds[layer].add_activations(mult=mult, start_pos=-1))
        #for prompt in tqdm(test_dataset, desc="Processing prompts"):
        for prompt in test_dataset:
            count += 1
            msg = prompt['question']
            a_matches = (prompt['answer_matching_behavior'] == '(A)')
            msg = "\x16user\n" + msg + "\x17\n\x16assistant\n("
            with torch.no_grad():
                out = model.forward(torch.tensor(tokenizer.encode(msg)).unsqueeze(0).cuda(device))

            logits = out.logits[0, -1, :]
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            a_prob = probs[a_token_id]
            b_prob = probs[b_token_id]
            
            max_prob = F.softmax(logits.float(), dim=-1).argmax().cpu().numpy()
            if max_prob == a_token_id and a_matches:
                correct+=1
            if max_prob == b_token_id and not a_matches:
                correct+=1
            #print(tokenizer.decode(np.atleast_1d(max_prob)),prompt['answer_matching_behavior'] )
            behavior_prob = a_prob if a_matches else b_prob
            average_prob += behavior_prob / (a_prob + b_prob)
            matching_prob += behavior_prob
            not_matching_prob += a_prob if not a_matches else b_prob
        results.append((layer,mult,-1,average_prob / count, matching_prob / count, not_matching_prob / count,correct/count))
        h.remove()
        print(f"{layer}, {mult}, {average_prob / count}, {matching_prob / count}, {not_matching_prob / count}, {correct/count}")


import matplotlib.pyplot as plt

import pandas as pd

cleandata=[]
colors = ["red","blue","green","yellow","purple","orange","black","pink","brown"]
for i,values in enumerate(results):
        cleandata.append([values[0],values[1],values[2],values[3][0],values[4][0],values[5][0],values[6]])
            
dataframe = pd.DataFrame(cleandata,columns=["Layer","Mult","Pos","Average","Matching","Not Matching","Correct"])
dataframe.to_csv("slimpj-dpo-survival-instincts.csv")

mults = dataframe["Mult"].unique()
for i,m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"]==-1]

    data_previous = data_previous[data_previous["Mult"]==m]
    plt.plot(data_previous["Layer"],data_previous["Matching"]/(data_previous["Matching"]+data_previous["Not Matching"]),color=colors[i],label=f"Mult: {m}")
plt.legend()
plt.figure()
mults = dataframe["Mult"].unique()
for i,m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"]==-1]

    data_previous = data_previous[data_previous["Mult"]==m]
    plt.plot(data_previous["Layer"],data_previous["Matching"],color=colors[i],label=f"Mult: {m}")
plt.legend()
plt.figure()
mults = dataframe["Mult"].unique()
for i,m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"]==-1]

    data_previous = data_previous[data_previous["Mult"]==m]
    plt.plot(data_previous["Layer"],data_previous["Not Matching"],color=colors[i],label=f"Mult: {m}")
plt.legend()

plt.figure()
mults = dataframe["Mult"].unique()
for i,m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"]==-1]
    if abs(m)>0.5:
        continue
    
    data_previous = data_previous[data_previous["Mult"]==m]
    plt.plot(data_previous["Layer"],data_previous["Not Matching"]+data_previous["Matching"],color=colors[i],label=f"Mult: {m}")
plt.legend()

plt.figure()

mults = dataframe["Mult"].unique()
for i,m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"]==-1]
    if abs(m)>0.5:
        continue
    data_previous = data_previous[data_previous["Mult"]==m]
    plt.plot(data_previous["Layer"],data_previous["Average"],color=colors[i],label=f"Mult: {m}")
plt.legend()



mults = dataframe["Mult"].unique()
for i,m in enumerate(mults):
    data_previous = dataframe[dataframe["Pos"]==-1]
    if abs(m)>0.5:
        continue
    data_previous = data_previous[data_previous["Mult"]==m]
    plt.plot(data_previous["Layer"],data_previous["Correct"],color=colors[i],label=f"Mult: {m}")
plt.legend()