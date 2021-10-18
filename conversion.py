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

def kg(embedding_path, conversion_path, output_path):
    embedding = pd.read_csv(embedding_path, header=None)
    conversion = pd.read_csv(conversion_path, header=None)
    embedding.index = embedding.index + 1
    embedding['movieID'] = -1
    # print(embedding)
    for index, row in embedding.iterrows():
        c = conversion.loc[conversion[0] == index]
        if len(c) == 1:
        	# print(c)
        	# print(c.iloc[0][1])
        	embedding.at[index,'movieID'] = c.iloc[0][1]
        	# print(row['movieID'])
    print(embedding['movieID'].value_counts())
    # embedding['movieID'] = embedding['movieID'].apply(np.int64)
    final = embedding.loc[embedding['movieID'] > -1]
    print(final)
    print(len(final))
    final.to_csv(output_path, header = False, index=False)

if __name__=='__main__':
    kg('data/embedding/embedding.csv', 'data/embedding/conversion.csv', 'data/embedding/kg.csv')
