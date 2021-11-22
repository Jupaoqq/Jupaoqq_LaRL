import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from latent_dialog import domain
from latent_dialog.metric import MetricsContainer
from latent_dialog.utils import read_lines
from latent_dialog.corpora import EOS, SEL
from latent_dialog import evaluators
import re


def movieID_to_embedding(movieID, kg):
    embed_df = kg.loc[kg[128] == int(movieID)]
    if not embed_df.empty:
        e = embed_df.iloc[:, 0:128].to_numpy()
        embed = th.as_tensor(e[0]).to(device="cuda").unsqueeze(0)
        return embed
    else:
        return ""

def softmax_func(z, profile, kg, w_matrix, use_z):
    # print("Dimensions:")
    # print(z.shape)
    # print(w_matrix.shape)
    e = ""
    movies = []
    # print(profile)
    for key, a in profile.items():
        embed = movieID_to_embedding(key, kg)
        # print(embed)
        if len(e) == 0:
            if len(embed) > 0:
                e = embed
                movies.append(key)
        else:
            if len(embed) > 0:
                e = th.cat([e, embed], dim = 0)
                movies.append(key)
                # print("Not found")
    # print(len(movies))
    # print(self.w_matrix)
    if len(e) > 0:
        if use_z:
            d = th.matmul(z, w_matrix)
            s = th.squeeze(d, 0)
            # print(s.shape)
            # print(e.shape)
            f = th.matmul(e.float(), th.transpose(s, 0, 1))
        else:
            # d = th.matmul(z, self.w_matrix)
            # s = th.squeeze(d, 0)
            f = th.matmul(e.float(), th.transpose(w_matrix, 0, 1))
        r = F.gumbel_softmax(f, dim=0)
        r_flat = th.flatten(r)
        # print(r_flat)

        _, ind_list = th.topk(r_flat, len(movies))

        embed_movie = th.index_select(e, 0, ind_list[0])
        

        ind = ind_list.tolist() 
        # print(movies)
        # print(ind)
        prob_sorted = (movies[i] for i in ind)    
        final = list(prob_sorted)
        # print(prob_sorted)
        
        true_labels = []
        for i in movies:
            pf = profile.get(i)
            if pf['liked'] == '1':
                true_labels.append(1)
            else:
                true_labels.append(0)

        t = th.as_tensor(true_labels).to(device="cuda")
        # print(t)
        # print(r_flat)
        loss = nn.BCELoss()(r_flat.float(), t.float())


        return embed_movie, final, loss
    else:
        # print("insufficient profile")
        return -1, [], -1