import json
import random

import numpy as np

import spacy
nlp = spacy.load('en_core_web_sm')


def tokenize(sent):
    doc = nlp(sent)
    return [token.lemma_ for token in doc if not token.is_punct]



def build_vocab(data_file):
    token_to_idx = dict()
    idx_to_token = dict()

    token_to_idx['{UNK}'] = 0

    with open(data_file, 'r') as f:
        intents = json.load(f)

    for intent in intents['intents']:
        patterns = intent['patterns']
        for pattern in patterns:
            words = tokenize(pattern)
            for word in words:
                if word not in token_to_idx:
                    token_to_idx[word] = len(token_to_idx)

    for (key,val) in token_to_idx.items():
        idx_to_token[val] = key
    
    return token_to_idx, idx_to_token

def build_label_vocab(data_file):
    token_to_idx = dict()
    idx_to_token = dict()

    with open(data_file, 'r') as f:
        intents = json.load(f)

    for intent in intents['intents']:
        tag = intent['tag']
        tag = tag.strip()
        if tag not in token_to_idx:
            token_to_idx[tag] = len(token_to_idx)
        

    for (key,val) in token_to_idx.items():
        idx_to_token[val] = key
    
    return token_to_idx, idx_to_token


def load_patterns(data_file, text_dict, label_dict):
    patterns_idx = []
    tag = []

    with open(data_file, 'r') as f:
        intents = json.load(f)

    for intent in intents['intents']:
        patterns = intent['patterns']
        label = intent['tag']
        for pattern in patterns:
            pattern_idx = []
            words = tokenize(pattern)
            for word in words:
                if word in text_dict:
                    pattern_idx.append(text_dict[word])
                else:
                    pattern_idx.append(text_dict['{UNK}'])
            patterns_idx.append(pattern_idx)
            tag.append(label_dict[label])
    return patterns_idx, tag

def prepare_test_sentence(sent, text_dict):
    words = tokenize(sent)
    pattern_idx = []
    for word in words:
        if word in text_dict:
            pattern_idx.append(text_dict[word])
        else:
            pattern_idx.append(text_dict['{UNK}'])
    return np.array(pattern_idx)

            



