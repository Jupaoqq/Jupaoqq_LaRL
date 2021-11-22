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
from latent_dialog.recommendation import softmax_func, movieID_to_embedding
from latent_dialog.main import get_sent
from sentence_transformers import SentenceTransformer, util
import scipy

class Dialog(object):
    """Dialogue runner."""

    def __init__(self, agents, args, usr_sent, sys_sent):
        assert len(agents) == 2
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self.rew = 0
        self.w_matrix = th.randn(256, 128, device='cuda')
        self.w_matrix_no_z = th.randn(1, 128, device='cuda')
        self.evaluator = evaluators.BleuEvaluator('Roll out')
        self._register_metrics()
        self.BERT_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.usr_embedding = self.BERT_model.encode(usr_sent)
        self.sys_embedding = self.BERT_model.encode(sys_sent)


    def consine_sim(self, queries, person):
        sentence_embeddings = None
        if person == "System":
            sentence_embeddings = self.sys_embedding
        else:
            sentence_embeddings = self.usr_embedding
        # queries = [query]
        # code adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py
        query_embeddings = self.BERT_model.encode(queries)

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        number_top_matches = 5 #@param {type: "number"}

        all_sim = 0
        count = 0
        for query, query_embedding in zip(queries, query_embeddings):
            distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            # print("\n\n======================\n\n")
            # print("Query:", query)
            # print("\nTop 5 most similar sentences in corpus:")

            total = 0
            for idx, distance in results[0:number_top_matches]:
                total = total + (1 - distance)
            all_sim = all_sim + (total / 5)
            count = count + 1

        return all_sim / count

    def _register_metrics(self):
        """Registers valuable metrics."""
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        # self.metrics.register_average('user_movie_mention')
        self.metrics.register_average('sys_movie_mention')
        self.metrics.register_average('sys_movie_mention_like')

        self.metrics.register_average('recall@1')
        self.metrics.register_average('recall@5')
        self.metrics.register_average('recall@10')

        # self.metrics.register_average('dist-1')
        self.metrics.register_average('dist-2')
        self.metrics.register_average('dist-3')
        self.metrics.register_average('dist-4')

        self.metrics.register_average('item ratio')

        for agent in self.agents:
            self.metrics.register_average('%s_rew' % agent.name)
            # self.metrics.register_percentage('%s_sel' % agent.name)
            # self.metrics.register_uniqueness('%s_unique' % agent.name)
        # text metrics
        ref_text = ' '.join(read_lines(self.args.ref_text))
        # self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return len(out) == 2 and out[0] == SEL and out[1] == EOS

    def show_metrics(self):
        return ' '.join(['%s = %s' % (k, v) for k, v in self.metrics.dict().items()])



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
        # assert len(self.agents) == len(ctxs)
        profile = ""
        # initialize agents by feeding in the context
        
        for agent, ctx in zip(self.agents, ctxs):
            agent.feed_context(ctx)
            if agent.name == "User":
                profile = agent.movie
            if agent.name == "System":
                # for name, param in agent.model.named_parameters():
                #     if param.requires_grad:
                #         print(name)
                #         print(param.data)
                self.w_matrix = agent.model.w_matrix
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
        sys_prob = []
        sys_sent = []

        first_turn = False
        nturn = 0
        s_turn = 0
        i_turn = 0

        # temp = ""
        while True:
            avg = th.zeros(128, device='cuda')
            nturn += 1
            # produce an utterance
            out, out_words, z, dec_last_s = writer.write() # out: list of word ids, int, len = max_words
            if verbose:
                print('\t{} out_words = {}'.format(writer.name, out_words))

            self.metrics.record('sent_len', len(out))
            # self.metrics.record('full_match', out_words)
            # self.metrics.record('%s_unique' % writer.name, out_words)

            to_analyze = []
            for ii in out_words:
                if "<" not in ii:
                    to_analyze.append(ii)
            rew_sent = self.consine_sim(["".join(to_analyze)], writer.name) 
            # print(rew_sent)
            # append the utterance to the conversation
            conv.append(out_words)
            rew = 0
            for i in out_words:
                if writer.name == "System":
                    s_turn += 1
                    
                    if "[ITEM]" in i:
                        i_turn += 1
                        w_m = self.w_matrix_no_z
                        use_z = False
                        if len(z) > 0:
                            w_m = self.w_matrix
                            use_z = True
                        embed, m_id, _ = softmax_func(z, profile, kg, w_m, use_z)
                        if len(m_id) > 0:
                            sys_prob.append(m_id)
                            rew = self.reward(profile, m_id[0])
                            if rew > 0.49:
                                sys_movie_mention_like.append(m_id[0])
                            history, mention, avg = self.mention(sys_movie_mention_history, sys_movie_mention, m_id, embed, user = False)
                            sys_movie_mention_history, sys_movie_mention = history, mention
                        else:
                            pass
                        
                # else:
                #     if "[ITEM]" in i:
                #         history, mention, avg = mention(user_movie_mention_history, user_movie_mention, i, user = True)
                #         user_movie_mention_history, user_movie_mention = history, mention
            if writer.name == "System":
                sys_reward.append(rew)
                sys_sent.append(out_words)
            reader.read(out, avg)

            # make the other agent to read it
            # check if the end of the conversation was generated

            if nturn > 15:
                break

            if self._is_selection(out_words) and first_turn:
                # self.metrics.record('%s_sel' % writer.name, 1)
                # self.metrics.record('%s_sel' % reader.name, 0)
                break
            writer, reader = reader, writer
            first_turn = True
            # if self.args.max_nego_turn > 0 and nturn >= self.args.max_nego_turn:
            #     return []

            # self.metrics.record('user_movie_mention', len(user_movie_mention))
            self.metrics.record('sys_movie_mention', len(sys_movie_mention))
            self.metrics.record('sys_movie_mention_like', len(sys_movie_mention_like))

        # print("sys reward:")
        # print(sys_reward)
        rewards[0] = sum(sys_reward)
        self.metrics.record('item ratio', i_turn/s_turn)
        # evaluate the choices, produce agreement and a reward
        # if verbose:
            # print('ctxs = {}'.format(ctxs))
            # print('task reward = {}'.format(rewards))



        if len(sys_prob) > 0:
            r1, r5, r10 = self.evaluator.rollout_recall(sys_prob[-1], profile)
            if r1 > -1:
                self.metrics.record('recall@1', r1)
            if r5 > -1:
                self.metrics.record('recall@5', r5)
            if r10 > -1:
                self.metrics.record('recall@10', r10)

        if len(sys_sent) > 0:
            d1, d2, d3, d4 = self.evaluator.distinct_metrics(sys_sent)
            self.metrics.record('dist-2', d2)
            self.metrics.record('dist-3', d3)
            self.metrics.record('dist-4', d4)

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
            # if agent.name == "System":
            self.metrics.record('%s_rew' % agent.name, reward)
        if verbose:
            # print("Profile:")
            # print(profile)
            print("Reward:")
            print(sys_reward)
            print('='*50)
            print(self.show_metrics())
            print('='*50)

        stats = dict()
        stats['recall@1'] = self.metrics.metrics['recall@1'].show()
        stats['recall@5'] = self.metrics.metrics['recall@5'].show()
        stats['recall@10'] = self.metrics.metrics['recall@10'].show()
        stats['dist-3'] = self.metrics.metrics['dist-3'].show()
        stats['system_rew'] = self.metrics.metrics['system_rew'].show()
        # stats['system_unique'] = self.metrics.metrics['system_unique'].show()

        return conv, rewards, stats


