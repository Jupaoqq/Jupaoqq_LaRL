import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy
import pandas as pd

def process(input, output1):
    li = []
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
        cl = []
        temp_id = 100000000000
        temp_sent = ""
        if (altitude and initial_altitude):
            for m in contexts:

                proc = m['text'].split()
                procced = []
                for token in proc:
                    if "@" in token:
                        token = "[ITEM]"
                    procced.append(token)
                newstr = " ".join(procced)

                temp_sent = temp_sent + " " + newstr

                if m['senderWorkerId'] != temp_id:
                    if temp_id == 100000000000:
                        temp_id = m['senderWorkerId']
                    else:
                        if m['senderWorkerId'] == seekerid and len(cl) > 0:
                            b = cl[:]
                            b.insert(0, temp_sent)
                            li.append(b)
                        cl.append(temp_sent)
                        temp_sent = ""
                        temp_id = m['senderWorkerId']

    df = pd.DataFrame(li)
    df.to_csv(output1, sep='|', encoding='utf-8', index=False)

if __name__=='__main__':
    process('data/raw/test_data.jsonl', 'aa.csv')
    # entity('data/raw/train_data.jsonl', 'data/raw/valid_data.jsonl', 'data/raw/test_data.jsonl', 'data/negotiate/entity.txt')
   