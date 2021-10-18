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


class Dialog(object):
    """Dialogue runner."""
    def __init__(self, agents, args):
        assert len(agents) == 2
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self.rew = 0
        self.w_matrix = th.randn(256, 128, device='cuda')
        self._register_metrics()

    def _register_metrics(self):
        """Registers valuable metrics."""
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_average('user_movie_mention')
        self.metrics.register_average('sys_movie_mention')
        self.metrics.register_average('sys_movie_mention_like')
        for agent in self.agents:
            self.metrics.register_average('%s_rew' % agent.name)
            self.metrics.register_percentage('%s_sel' % agent.name)
            self.metrics.register_uniqueness('%s_unique' % agent.name)
        # text metrics
        ref_text = ' '.join(read_lines(self.args.ref_text))
        self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return len(out) == 2 and out[0] == SEL and out[1] == EOS

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def movieID_to_embedding(self, movieID, kg):
        embed_df = kg.loc[kg[128] == int(movieID)]
        if not embed_df.empty:
            e = embed_df.iloc[:, 0:128].to_numpy()
            embed = th.as_tensor(e[0]).to(device="cuda").unsqueeze(0)
            return embed
        else:
            return ""

    def softmax_func(self, z, profile, kg):
        # print("Dimensions:")
        # print(z)
        # print(z.shape)
        e = ""
        movies = []
        # print(profile)
        for key, a in profile.items():
            embed = self.movieID_to_embedding(key, kg)
            # print(embed)
            if len(e) == 0:
                e = embed
                movies.append(key)
            else:
                try:
                    e = th.cat([e, embed], dim = 0)
                    movies.append(key)
                except:
                    pass
                    # print("Not found")
        # print(movies)
        # print(self.w_matrix)
        d = th.matmul(z, self.w_matrix)
        s = th.squeeze(d, 0)
        f = th.matmul(e.float(), th.transpose(s, 0, 1))
        # print(f)
        r = F.softmax(f, dim=0)
        r_flat = th.flatten(r)
        # print(r_flat)
        _, ind_list = th.topk(r_flat, 1)
        # print(ind_list)
        ind = ind_list.tolist()[0] 
        embed_movie = th.index_select(e, 0, ind_list)

        # Other reward func?      
        
        return embed_movie, movies[ind]

    def mention(self, history, mention, id, embed, user):
        if len(history) == 0:
            history = embed
        else:
            history = th.cat([history, embed], dim = 0)
        # print(history.shape)
        # print(history)
       
        mention.append(id)
        
        # if len(mention) != 0:
        sum = history.sum(dim=0)
        avg = th.div(sum, len(mention)).unsqueeze(0).float()
        return history, mention, avg

    def reward(self, profile, movieID):
        rew = 0
        for key, value in profile.items():
            if key == movieID:

                if value['seen'] == '1':
                    rew = 0.05
                if value['liked'] == '1':
                    rew = 0.5
                if value['liked'] == '0':
                    rew = -0.5

        return rew


    def run(self, ctxs, kg, update = True, verbose=True):
        """Runs one instance of the dialogue."""
        assert len(self.agents) == len(ctxs)
        # initialize agents by feeding in the context
        for agent, ctx in zip(self.agents, ctxs):
            agent.feed_context(ctx)
            if agent.name == "User":
                profile = agent.movie
        sys_reward = []
        rewards = [0,0]
        # choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        begin_name = writer.name
        if verbose:
            print('begin_name = {}'.format(begin_name))

        # initialize BOD utterance for each agent
        writer.bod_init('writer')
        reader.bod_init('reader')

        conv = []
        # reset metrics
        self.metrics.reset()
        sys_movie_mention_history = ""
        # print("sys")
        # print(sys_movie_mention_history)
        # user_movie_mention_history = th.zeros(128, device='cuda')

        user_movie_mention = []
        sys_movie_mention = []
        sys_movie_mention_like = []

        first_turn = False
        nturn = 0


        # temp = ""
        while True:
            avg = th.zeros(128, device='cuda')
            nturn += 1
            # produce an utterance
            out, out_words, z = writer.write() # out: list of word ids, int, len = max_words
            if verbose:
                print('\t{} out_words = {}'.format(writer.name, out_words))

            self.metrics.record('sent_len', len(out))
            self.metrics.record('full_match', out_words)
            self.metrics.record('%s_unique' % writer.name, out_words)

            # append the utterance to the conversation
            conv.append(out_words)
            rew = 0
            for i in out_words:
                if writer.name == "System": 
                    
                    if "[ITEM]" in i:
                        embed, m_id = self.softmax_func(z, profile, kg)
                        rew = self.reward(profile, m_id)
                        history, mention, avg = self.mention(sys_movie_mention_history, sys_movie_mention, m_id, embed, user = False)
                        sys_movie_mention_history, sys_movie_mention = history, mention
                        
                # else:
                #     if "[ITEM]" in i:
                #         history, mention, avg = mention(user_movie_mention_history, user_movie_mention, i, user = True)
                #         user_movie_mention_history, user_movie_mention = history, mention
            if writer.name == "System":
                sys_reward.append(rew)
            reader.read(out, avg)

            # make the other agent to read it
            # check if the end of the conversation was generated

            if nturn > 15:
                break

            if self._is_selection(out_words) and first_turn:
                self.metrics.record('%s_sel' % writer.name, 1)
                self.metrics.record('%s_sel' % reader.name, 0)
                break
            writer, reader = reader, writer
            first_turn = True
            if self.args.max_nego_turn > 0 and nturn >= self.args.max_nego_turn:
                return []

            # self.metrics.record('user_movie_mention', len(user_movie_mention))
            self.metrics.record('sys_movie_mention', len(sys_movie_mention))
            self.metrics.record('sys_movie_mention_like', len(sys_movie_mention_like))

        # print("sys reward:")
        # print(sys_reward)
        # rewards[0] = sys_reward 
        # evaluate the choices, produce agreement and a reward
        # if verbose:
            # print('ctxs = {}'.format(ctxs))
            # print('task reward = {}'.format(rewards))

        # perform update, in case if any of the agents is learnable
        if update:
            for agent, reward in zip(self.agents, rewards):
                if agent.name == "System":
                    if not all(v == 0 for v in sys_reward):
                        agent.update(sys_reward)
                    else:
                        pass
                        # print("no update")

        self.metrics.record('dialog_len', len(conv))
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record('%s_rew' % agent.name, np.average(reward))
        if verbose:
            # print("Profile:")
            # print(profile)
            print("Reward:")
            print(sys_reward)
            print('='*50)
            print(self.show_metrics())
            print('='*50)

        stats = dict()
        stats['system_rew'] = self.metrics.metrics['system_rew'].show()
        stats['system_unique'] = self.metrics.metrics['system_unique'].show()

        return conv, rewards, stats


