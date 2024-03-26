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


def latent_state_reading(source_phrase,eliciting_phrases,model,tokenizer,multiplyer=10):
    phrase = tokenizer.encode(source_phrase,return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.forward(phrase,cache=True)
    logits = outputs.logits
    chosen = torch.argmax(logits, dim=-1)
    #print(tokenizer.batch_decode(chosen))
    source_cache = deepcopy(outputs.cache_params)
    predictions={}
    print("Doing source phrase: ",source_phrase)
    for phrase in eliciting_phrases:
        original_phrase = phrase
        print("Doing phrase: ",original_phrase)
        predictions[original_phrase]={}
        phrase_ids = tokenizer.encode(phrase,return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.forward(phrase_ids[:,:-1],cache=True)
        logits = outputs.logits
        chosen = torch.argmax(logits, dim=-1)
        #print(tokenizer.batch_decode(chosen))
        eliciting_cache = deepcopy(outputs.cache_params)

        new_cache = eliciting_cache
        multiplyers = [0,multiplyer]
        for multiplyer in multiplyers:

            for i in range(len(new_cache.ssm_states)):
                new_cache.ssm_states[i] += multiplyer*(source_cache.ssm_states[i])
           #     new_cache.conv_states[i] += multiplyer*(source_cache.conv_states[i])
            string=""
            used_cache = deepcopy(new_cache)
            phrase = phrase_ids[:,-1].unsqueeze(0)
            for j in range(4):
                with torch.no_grad():
                    outputs = model.forward(phrase,cache_params=used_cache)
                logits = outputs.logits
                chosen = torch.argmax(logits, dim=-1)
                used_cache = deepcopy(outputs.cache_params)
                string+=tokenizer.decode(chosen[0])
                phrase = chosen

            predictions[original_phrase][multiplyer]=string
        print(predictions[original_phrase])
    return predictions

def layer_latent_state_reading(source_phrase,eliciting_phrase,model,tokenizer,layer=0,multiplyer=10):
    phrase = tokenizer.encode(source_phrase,return_tensors="pt").to("cuda")
    cache_params=None
    with torch.no_grad():
        outputs = model.forward(phrase,cache=True,cache_params=None
    )
    logits = outputs.logits
    source_cache = deepcopy(outputs.cache_params)
    phrase_ids = tokenizer.encode(eliciting_phrase,return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.forward(phrase_ids[:,:-1],cache=True)
    logits = outputs.logits
    eliciting_cache = deepcopy(outputs.cache_params,cache_params=None
    )
    new_cache = eliciting_cache
    new_cache.ssm_states[layer] += multiplyer*(source_cache.ssm_states[layer])
    #new_cache.conv_states[layer] += multiplyer/10*(source_cache.conv_states[layer])
    string=""
    used_cache = deepcopy(new_cache)
    phrase = phrase_ids[:,-1].unsqueeze(0)
    for j in range(5):
        with torch.no_grad():
            outputs = model.forward(phrase,cache_params=used_cache)
        logits = outputs.logits
        chosen = torch.argmax(logits, dim=-1)
        used_cache = deepcopy(outputs.cache_params)
        string+=tokenizer.decode(chosen[0])
        phrase = chosen

    return string

def latent_state_decoding(source_phrase,state,model,tokenizer,multiplyer=10):
    phrase_ids = tokenizer.encode(source_phrase,return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.forward(phrase_ids[:,:-1],cache=True,cache_params=None
    )
    new_cache = deepcopy(outputs.cache_params)
    for i in range(len(new_cache.ssm_states)):
        new_cache.ssm_states[i] += multiplyer*(state.ssm_states[i])
    #new_cache.conv_states[layer] += multiplyer/10*(source_cache.conv_states[layer])
    string=""
    used_cache = deepcopy(new_cache)
    phrase = phrase_ids[:,-1].unsqueeze(0)
    for j in range(5):
        with torch.no_grad():
            outputs = model.forward(phrase,cache_params=used_cache)
        logits = outputs.logits
        chosen = torch.argmax(logits, dim=-1)
        used_cache = deepcopy(outputs.cache_params)
        string+=tokenizer.decode(chosen[0])
        phrase = chosen

    return string

def get_average_state(phrases,model,tokenizer):
    phrase = tokenizer.encode(phrases[0],return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.forward(phrase,cache=True)
    source_cache = deepcopy(outputs.cache_params)
    for phrase in phrases[1:]:
        phrase_ids = tokenizer.encode(phrase,return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.forward(phrase_ids,cache=True)
        eliciting_cache = deepcopy(outputs.cache_params)
        for i in range(len(source_cache.ssm_states)):
            source_cache.ssm_states[i] += eliciting_cache.ssm_states[i]
            #source_cache.conv_states[i] += eliciting_cache.conv_states[i]
    for i in range(len(source_cache.ssm_states)):
        source_cache.ssm_states[i] /= len(phrases)
        #source_cache.conv_states[i] /= len(phrases)
    return source_cache


eliciting_phrases = [" ","I'm thinking about the country of","I'm thinking about the city of","I was built in the year of","dog -> dog; cat -> cat; object -> object; animal -> animal; human -> human;"]

model = convert_mamba_to_hf("state-spaces/mamba-2.8b-slimpj")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = model.to("cuda")



source_phrases = ["Eiffel Tower",
                  "Spain",
                  "The Colosseum",
                  "The sun",
                  "An apple", 
                  "A plate of pasta"
                  ]
eliciting_phrases = ["Repeat: Apple. Apple. Repeat: World. World. Repeat: Dog. Dog. Repeat: Cat. Cat. Repeat:",
                     "The country of",
                     "The city of",
                    # "In the year of",
                    #  "The language of",
                    #  "In the continent of",
                    #  "This food is",
                    #  "That color is",
                    #  ":\n",
                     ]
#eliciting_phrases = ["I'm thinking about the country of"]

for source_phrase in source_phrases:
    predictions = latent_state_reading(source_phrase,eliciting_phrases,model,tokenizer,1)        


for i in range(64):
    print("Layer: ",i)
    print(layer_latent_state_reading("The Colosseum","The country of",model,tokenizer,i,25))

monuments = ["The Eiffel Tower",
             "The Colosseum",
             "The Statue of Liberty",
             "The Great Wall of China",
             "The Pyramids of Giza",
             "The Taj Mahal",
             "The Leaning Tower of Pisa",
             "The Sydney Opera House",
             "The Parthenon",
             "The Great Sphinx of Giza",
             "The Moai Statues of Easter Island",
             "The Christ the Redeemer Statue",
             "The Acropolis of Athens",
             "The Alhambra",
             "The Angkor Wat",
             "The Burj Khalifa",
             "The Chichen Itza",
             "The Hagia Sophia",
             "The Machu Picchu",
             "The Petra"]

for source_phrase in monuments:
    latent_state_reading(source_phrase,eliciting_phrases,model,tokenizer,1.5)        

cities = ["City",
            "Metropolis",
            "I'm thinking of a city",
            "The city of",
            "Is a city",
            "The city of Lisbon",
            "I live in the city of",
            "I was born in the city of",
            "I'm visiting the city of",
            "This monument is in the city of",]
cities = ["This monument is in the city of"]

average_city_state = get_average_state(cities,model,tokenizer)
for source_phrase in monuments:
    print(latent_state_decoding(source_phrase,average_city_state,model,tokenizer,0.5)  )


