import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy


def process(input, output, dummy = True):
    f = open(input, encoding='utf-8')
    f2 = open(output, "a", encoding='utf-8')
    for line in tqdm(f):
        lines=json.loads(line.strip())
        seekerid=lines["initiatorWorkerId"]
        recommenderid=lines["respondentWorkerId"]
        contexts=lines['messages']
        movies=lines['movieMentions']
        altitude=lines['respondentQuestions']
        initial_altitude=lines['initiatorQuestions']
        l = ""
        if (altitude and initial_altitude):
            last = ""
            l += "<dialogue> "
            temp_id = 100000000000
            for m in contexts:
                if m['senderWorkerId'] != temp_id:
                    if temp_id != 100000000000:
                        l += " <eos> "
                    temp_id = m['senderWorkerId']
                    if m['senderWorkerId'] == seekerid:
                        l += "YOU: "
                        last = "THEM: <selection> "
                    elif m['senderWorkerId'] == recommenderid:
                        l += "THEM: "
                        last = "YOU: <selection> "
                    else:
                        pass
                l += m['text']      
                l += " "
            l += " <eos> "
            l += last
            l += "</dialogue> " 
            l += " <user> "
            for key, a in altitude.items():
                l += ("%s %s %s %s " % (key, a['suggested'], a['seen'], a['liked']))
            l += "</user> "
            l = l.replace('\r', '')
            l = l.replace('\n', '')
            l += "\n"
            # print(l)
            f2.write(l)

def selfplay(input, output):
    f = open(input, encoding='utf-8')
    f2 = open(output, "a", encoding='utf-8')
    user = dict()
    for line in tqdm(f):
        lines = json.loads(line.strip())
        seekerid = lines["initiatorWorkerId"]
        initial_altitude = lines['initiatorQuestions']
        if initial_altitude: 
            if seekerid in user:
                user[seekerid].update(initial_altitude)
            else:
                user[seekerid] = initial_altitude
            # for key, a in initial_altitude.items():c
            #     l += (" %s %s %s" % (key, a['seen'], a['liked']))
            # l += "\n"
            # print(l)
            # f2.write(l)
    for key1, a in user.items():
        l = ""
        # l += str(key1)
        for key2, b in a.items():
            l += (" %s %s %s" % (key2, b['seen'], b['liked']))
        l += "\n"
        # print(l)
        f2.write(l)

def entity(input1, input2, input3, output):
    f1 = open(input1, encoding='utf-8')
    f2 = open(input2, encoding='utf-8')
    f3 = open(input3, encoding='utf-8')
    f4 = open(output, "a", encoding='utf-8')
    temp = []

    for line in tqdm(f1):
        lines=json.loads(line.strip())
        initial_altitude=lines['initiatorQuestions']
        if initial_altitude: 
            for key, a in initial_altitude.items():
                if key not in temp:
                    temp.append(key)
    for line in tqdm(f2):
        lines=json.loads(line.strip())
        initial_altitude=lines['initiatorQuestions']
        if initial_altitude: 
            for key, a in initial_altitude.items():
                if key not in temp:
                    temp.append(key)
    for line in tqdm(f3):
        lines=json.loads(line.strip())
        initial_altitude=lines['initiatorQuestions']
        if initial_altitude: 
            for key, a in initial_altitude.items():
                if key not in temp:
                    temp.append(key)

    for t in temp:
        f4.write(t + "\n")

if __name__=='__main__':
    process('data/raw/train_data.jsonl', 'data/negotiate/train.txt')
    process('data/raw/valid_data.jsonl', 'data/negotiate/val.txt')
    process('data/raw/test_data.jsonl', 'data/negotiate/test.txt')
    # entity('data/raw/train_data.jsonl', 'data/raw/valid_data.jsonl', 'data/raw/test_data.jsonl', 'data/negotiate/entity.txt')
    selfplay('data/raw/train_data.jsonl', 'data/negotiate/selfplay.txt')
    selfplay('data/raw/valid_data.jsonl', 'data/negotiate/selfplay_eval.txt')