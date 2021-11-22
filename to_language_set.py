import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy


def process(input, output1, output2, dummy = True):
    f = open(input, encoding='utf-8')
    text_usr = []
    text_sys = []
    for line in tqdm(f):
        lines=json.loads(line.strip())
        seekerid=lines["initiatorWorkerId"]
        recommenderid=lines["respondentWorkerId"]
        contexts=lines['messages']
        altitude=lines['respondentQuestions']
        initial_altitude=lines['initiatorQuestions']
        if (altitude and initial_altitude):
            for m in contexts:
                proc = m['text'].split()
                procced = []
                for token in proc:
                    if "@" in token:
                        token = "[ITEM]"
                    procced.append(token)
                newstr = " ".join(procced)
                if m['senderWorkerId'] == seekerid:
                    text_usr.append(newstr)
                elif m['senderWorkerId'] == recommenderid:
                    text_sys.append(newstr)
    # print(text_sys)
    textfile1= open(output1, "w", encoding='utf-8')
    for element1 in text_usr:
        textfile1.write(element1 + "\n")
    textfile1.close()
    textfile2 = open(output2, "w", encoding='utf-8')
    for element2 in text_sys:
        textfile2.write(element2 + "\n")
    textfile2.close()
if __name__=='__main__':
    process('data/raw/train_data.jsonl', 'data/similarity/user.txt', 'data/similarity/system.txt')
    # entity('data/raw/train_data.jsonl', 'data/raw/valid_data.jsonl', 'data/raw/test_data.jsonl', 'data/negotiate/entity.txt')
   