import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy


def process(input1, input2, output):
    f = open(input1, encoding='utf-8')
    f2 = open(input2, encoding='utf-8')
    mdata = []
    for line in tqdm(f):
        lines=json.loads(line.strip())
        data = {}
        data['conv_id'] = lines['conversationId']
        data['profile'] = lines['respondentQuestions']
        mdata.append(data)
    adata = json.load(f2)
    # print(train_data)
    fdata = []
    for conv in tqdm(adata):
        data2 = conv
        prof = ""
        for profile in mdata:
            if profile['conv_id'] == data2['conv_id']:
                prof = profile['profile']
                
        data2['profile'] = prof
        # print(data2)
        if len(prof) != 0:
            fdata.append(data2)

    with open(output, 'w') as f:
        json.dump(fdata, f)
    



if __name__=='__main__':
    process('data/raw/train_data.jsonl', 'data/other/train_data.json', 'data/final/train_data.json')
    process('data/raw/valid_data.jsonl', 'data/other/valid_data.json', 'data/final/valid_data.json')
    process('data/raw/test_data.jsonl','data/other/test_data.json', 'data/final/test_data.json')
