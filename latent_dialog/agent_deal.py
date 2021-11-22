import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from latent_dialog import domain
from latent_dialog.utils import LONG, FLOAT
from latent_dialog.corpora import USR, SYS, BOD, EOS, STOP_TOKENS, SEL


class Agent(object):
    """Agent's interface.

    The dialogue should proceed in the following way:

    1) feed_context to each of the agent.
    2) randomly pick an agent who will start the conversation.
    3) the starting agent will write down her utterance.
    4) the other agent will read the pronounced utterance.
    5) unless the end of dialogue is pronounced, swap the agents and repeat the steps 3-4.
    6) once the conversation is over, generate choices for each agent and calculate the reward.
    7) pass back to the reward to the update function.


    See Dialogue.run in the dialog.py for more details.
    """

    def feed_context(self, context):
        """Feed context in to start new conversation.

        context: a list of context tokens.
        """
        pass

    def read(self, kg, inpt):
        """Read an utterance from your partner.

        inpt: a list of English words describing a sentence.
        """
        pass

    def write(self):
        """Generate your own utterance."""
        pass

    def choose(self):
        """Call it after the conversation is over, to make the selection."""
        pass

    def update(self, reward):
        """After end of each dialogue the reward will be passed back to update the parameters.

        agree: a boolean flag that specifies if the agents agreed on the deal.
        reward: the reward that the agent receives after the dialogue. 0 if there is no agreement.
        """
        pass


class LstmAgent(Agent):
    """An agent that uses DialogModel as an AI."""
    def __init__(self, model, corpus, args, name):
        super(LstmAgent, self).__init__()
        self.model = model
        self.corpus = corpus
        self.args = args
        self.name = name
        self.domain = domain.get_domain(args.domain)
        self.context = None
        self.goal_h = None
        self.dlg_history = None
        # self.lang_is = None
        self.lang_os = None
        self.lang_h = None

    def feed_context(self, context):
        # print(context)
        self.context = context
        movie_temp = context
        movie = dict()
        for i in range(len(movie_temp)//3):
            movie[movie_temp[i*3]] = {}
            movie[movie_temp[i*3]]['seen'] = movie_temp[i*3+1]
            movie[movie_temp[i*3]]['liked'] = movie_temp[i*3+2]
        self.movie = movie
        # print(movie)
        # print(self.movie)
        # context_id = np.array(self.corpus.goal2id(self.context))
        # context_var = self.model.np2var(context_id, LONG).unsqueeze(0) # (1, goal_len)
        # self.goal_h = self.model.goal_encoder(context_var) # (1, goal_nhid)
        self.goal_h = th.zeros((1, 128), device='cuda')

        # data that need updating and collecting
        self.dlg_history = [] # len = max_dlg_len
        # self.lang_is = [] # max_dlg_len*(1, 1, num_direction*utt_cell_size)
        self.lang_os = [] # max_dlg_len*(1, 1, dlg_cell_size)
        self.lang_h = None # tuple: (h, c)


        self.token_embed = [] # max_dlg_len*(1, max_utt_len, embedding_dim+1)
        self.token_feat  = [] # max_dlg_len*(1, max_utt_len, num_directions*utt_cell_size)
        self.turn_feat   = [] # max_dlg_len*(1, max_utt_len, dlg_cell_size)

    def bod_init(self, mission):
        if mission == 'writer':
            SPEAKER = USR
        elif mission == 'reader':
            SPEAKER = SYS
        else:
            print('Invalid mission')
            exit(-1)
        init_turn = self.corpus.sent2id([SPEAKER, BOD, EOS])
        self.read(init_turn, self.goal_h, require_speaker=False)

    def read(self, inpt, kg, require_speaker=True):
        # print(kg)
        # print(th.all(th.eq(kg, self.goal_h)))
        # print(kg.shape)
        # if len(kg) != 0 and not th.all(th.eq(kg, self.goal_h)):
        #     self.goal_h = kg
        inpt = self.corpus.sent2id([USR]) + inpt if require_speaker else inpt
        self.dlg_history.append(inpt)
        inpt_var = self.model.np2var(np.array(inpt), LONG).view(1, 1, len(inpt)) # (1, 1, max_words+1)
        feats = self.model.np2var(np.ones((1, 1)), FLOAT) # (1, 1)
        # enc_inputs: (1, 1, num_directions*utt_cell_size)
        # token_embed: (1, 1*max_utt_len, embedding_dim+1)
        # token_feat: (1, 1*max_utt_len, num_directions*utt_cell_size)
        enc_inputs, token_embed, token_feat = self.model.utt_encoder(inpt_var, feats=feats, goals=self.goal_h)
        self.token_embed.append(token_embed)
        self.token_feat.append(token_feat)
        # self.lang_is.append(enc_inputs)
        enc_outputs, self.lang_h = self.model.ctx_encoder(enc_inputs, init_state=self.lang_h, goals=None) # enc_outputs: (1, 1, dlg_cell_size)
        turn_feat = enc_outputs.unsqueeze(2).repeat(1, 1, inpt_var.size(-1), 1).view(enc_outputs.size(0), -1, enc_outputs.size(2)) # (1, 1*max_utt_len, dlg_cell_size)
        self.turn_feat.append(turn_feat)
        self.lang_os.append(enc_outputs)
        # assert (th.cat(self.lang_is, 1).size(1) == th.cat(self.lang_os, 1).size(1) and th.cat(self.lang_is, 1).size(1) == len(self.dlg_history))
        assert len(set([th.cat(self.token_embed, 1).size(1), th.cat(self.token_feat, 1).size(1), th.cat(self.turn_feat, 1).size(1)])) == 1 and len(set([len(self.dlg_history), len(self.lang_os), len(token_embed), len(token_feat), len(turn_feat)]))

    def write(self, max_words=None, stop_tokens=STOP_TOKENS):
        max_words = self.args.max_words if max_words is None else max_words
        inpt = self.corpus.sent2id([SYS])
        inpt_var = self.model.np2var(np.array(inpt), LONG).view(1, len(inpt))
        _, outs, dec_last_s = self.model.decoder.write(inpt_var, self.lang_h, self.lang_os, max_words, self.model.vocab, stop_tokens, self.goal_h)
        if outs[-1] == self.corpus.sent2id([SEL])[-1]:
            eos_patch = self.corpus.sent2id([EOS])
            outs += eos_patch
        inpt += outs
        self.read(inpt, self.goal_h, require_speaker=False)
        return outs, self.corpus.id2sent(outs), '', dec_last_s

    def transform_dialogue_history(self):
        self.dialogue_text = []
        for turn in self.dlg_history:
            self.dialogue_text.append(self.corpus.id2sent(turn))
        return self.dialogue_text 

class RlAgent(LstmAgent):
    """An Agent that updates the model parameters using REINFORCE to maximize the reward."""
    def __init__(self, model, corpus, args, name):
        super(RlAgent, self).__init__(model, corpus, args, name)
        self.opt = optim.SGD(
            self.model.parameters(),
            lr=self.args.rl_lr,
            momentum=self.args.momentum,
            nesterov=(self.args.nesterov and self.args.momentum > 0))
        # self.opt = optim.Adam(self.model.parameters(), lr=0.01)
        # self.opt = optim.RMSprop(self.model.parameters(), lr=0.0005)
        self.all_rewards = []
        self.model.train()

    def feed_context(self, ctx):
        super(RlAgent, self).feed_context(ctx)
        # save all the log probs for each generated word,
        # so we can use it later to estimate policy gradient.
        self.logprobs = [] # put this variable here because we want to clear it at the beginning of each episode

    def write(self, max_words=None, stop_tokens=STOP_TOKENS):
        max_words = self.args.max_words if max_words is None else max_words
        inpt = self.corpus.sent2id([SYS])
        inpt_var = self.model.np2var(np.array(inpt), LONG).view(1, len(inpt))
        logprobs, outs, dec_last_s = self.model.decoder.write(inpt_var, self.lang_h, self.lang_os, max_words, self.model.vocab, stop_tokens, self.goal_h)
        self.logprobs.extend(logprobs)
        if outs[-1] == self.corpus.sent2id([SEL])[-1]:
            eos_patch = self.corpus.sent2id([EOS])
            outs += eos_patch
        inpt += outs
        self.read(inpt, self.goal_h, require_speaker=False)
        return outs, self.corpus.id2sent(outs), '', dec_last_s

    def update(self, reward):
        reward_ = sum(reward)
        self.all_rewards.append(reward_)
        # standardize the reward
        r = (reward_ - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        # compute accumulated discounted reward
        g = self.model.np2var(np.array([r]), FLOAT).view(1, 1)
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.rl_clip)
        self.opt.step()


class LatentAgent(Agent):
    """An agent that uses DialogModel as an AI."""
    def __init__(self, model, corpus, args, name):
        super(LatentAgent, self).__init__()
        self.model = model
        self.corpus = corpus
        self.args = args
        self.name = name
        self.domain = domain.get_domain(args.domain)
        self.context = None
        self.goal_h = None
        self.dlg_history = None
        # self.lang_is = Nonec
        self.lang_os = None
        self.lang_h = None
        

    def feed_context(self, context):
        # print(context)
        self.context = context
        movie_temp = context
        # print("hello")
        # print(movie_temp)
        # print("hello2")
        # print(len(movie_temp)//3)

        movie = dict()
        for i in range(len(movie_temp)//3):
            # print(movie_temp[i*3])
            movie[movie_temp[i*3]] = {}
            movie[movie_temp[i*3]]['seen'] = movie_temp[i*3+1]
            movie[movie_temp[i*3]]['liked'] = movie_temp[i*3+2]
        self.movie = movie
        # print("hello3")
        # print(movie)
        self.goal_h = th.zeros((1, 128), device='cuda')

        # data that need updating and collecting
        self.dlg_history = [] # len = max_dlg_len
        # self.lang_is = [] # max_dlg_len*(1, 1, num_direction*utt_cell_size)
        self.lang_os = [] # max_dlg_len*(1, 1, dlg_cell_size)
        self.lang_h = None # tuple: (h, c)

        self.token_embed = [] # max_dlg_len*(1, max_utt_len, embedding_dim+1)
        self.token_feat  = [] # max_dlg_len*(1, max_utt_len, num_directions*utt_cell_size)
        self.turn_feat   = [] # max_dlg_len*(1, max_utt_len, dlg_cell_size)

    def bod_init(self, mission):
        if mission == 'writer':
            SPEAKER = USR
        elif mission == 'reader':
            SPEAKER = SYS
        else:
            print('Invalid mission')
            exit(-1)
        init_turn = self.corpus.sent2id([SPEAKER, BOD, EOS])
        self.read(init_turn, self.goal_h, require_speaker=False)

    def read(self, inpt, kg, require_speaker=True):
        # if len(kg) != 0 and not th.all(th.eq(kg, self.goal_h)):
        #     self.goal_h = kg
        inpt = self.corpus.sent2id([USR]) + inpt if require_speaker else inpt
        self.dlg_history.append(inpt)
        inpt_var = self.model.np2var(np.array(inpt), LONG).view(1, 1, len(inpt)) # (1, 1, max_words+1)
        # enc_inputs: (1, 1, num_directions*utt_cell_size)
        # token_embed: (1, 1*max_utt_len, embedding_dim+1)
        # token_feat: (1, 1*max_utt_len, num_directions*utt_cell_size)
        enc_inputs, token_embed, token_feat = self.model.utt_encoder(inpt_var, goals=self.goal_h)
        self.token_embed.append(token_embed)
        self.token_feat.append(token_feat)
        # self.lang_is.append(enc_inputs)
        enc_outputs, self.lang_h = self.model.ctx_encoder(enc_inputs, init_state=self.lang_h, goals=None) # enc_outputs: (1, 1, dlg_cell_size)
        turn_feat = enc_outputs.unsqueeze(2).repeat(1, 1, inpt_var.size(-1), 1).view(enc_outputs.size(0), -1, enc_outputs.size(2)) # (1, 1*max_utt_len, dlg_cell_size)
        self.turn_feat.append(turn_feat)
        self.lang_os.append(enc_outputs)
        # assert (th.cat(self.lang_is, 1).size(1) == th.cat(self.lang_os, 1).size(1) and th.cat(self.lang_is, 1).size(1) == len(self.dlg_history))
        assert len(set([th.cat(self.token_embed, 1).size(1), th.cat(self.token_feat, 1).size(1), th.cat(self.turn_feat, 1).size(1)])) == 1 \
               and len(set([len(self.dlg_history), len(self.lang_os), len(token_embed), len(token_feat), len(turn_feat)]))

    def transform_dialogue_history(self):
        self.dialogue_text = []
        for turn in self.dlg_history:
            self.dialogue_text.append(self.corpus.id2sent(turn))
        return self.dialogue_text

    def write(self, max_words=None, stop_tokens=STOP_TOKENS):
        max_words = self.args.max_words if max_words is None else max_words
        inpt = self.corpus.sent2id([SYS])
        inpt_var = self.model.np2var(np.array(inpt), LONG).view(1, len(inpt))
        dec_init_s, attn_ctxs, logprob_z = self.model.z2dec(self.lang_h, requires_grad=False)

        _, outs, dec_last_s = self.model.decoder.write(inpt_var, dec_init_s, attn_ctxs, max_words, self.model.vocab, stop_tokens, self.goal_h)
        if outs[-1] == self.corpus.sent2id([SEL])[-1]:
            eos_patch = self.corpus.sent2id([EOS])
            outs += eos_patch
        inpt += outs
        self.read(inpt, self.goal_h, require_speaker=False)
        return outs, self.corpus.id2sent(outs), dec_init_s , dec_last_s


class LatentRlAgent(LatentAgent):
    """An Agent that updates the model parameters using REINFORCE to maximize the reward."""
    def __init__(self, model, corpus, args, name, use_latent_rl=True):
        super(LatentRlAgent, self).__init__(model, corpus, args, name)
        self.opt = optim.SGD(
            self.model.parameters(),
            lr=self.args.rl_lr,
            momentum=self.args.momentum,
            nesterov=(self.args.nesterov and self.args.momentum > 0))
        # self.opt = optim.Adam(self.model.parameters(), lr=0.01)
        # self.opt = optim.RMSprop(self.model.parameters(), lr=0.0005)
        self.use_latent_rl = use_latent_rl
        self.all_rewards = []
        self.model.train()

    def feed_context(self, ctx):
        super(LatentRlAgent, self).feed_context(ctx)
        # save all the log probs for each generated word,
        # so we can use it later to estimate policy gradient.
        self.logprobs = [] # put this variable here because we want to clear it at the beginning of each episode

    def write(self, max_words=None, stop_tokens=STOP_TOKENS):
        max_words = self.args.max_words if max_words is None else max_words
        inpt = self.corpus.sent2id([SYS])
        inpt_var = self.model.np2var(np.array(inpt), LONG).view(1, len(inpt))
        if self.use_latent_rl:
            dec_init_s, attn_ctxs, logprob_z = self.model.z2dec(self.lang_h, requires_grad=False)
            logprobs, outs, dec_last_s = self.model.decoder.write(inpt_var, dec_init_s, self.lang_os, max_words, self.model.vocab, stop_tokens, self.goal_h)
            # print("logprobs")
            # print(logprob_z)
            self.logprobs.append(logprob_z)
            # print("logprobs_full")
            # print(self.logprobs)
        else:
            dec_init_s, attn_ctxs, logprob_z = self.model.z2dec(self.lang_h, requires_grad=True)
            logprobs, outs, dec_last_s = self.model.decoder.write(inpt_var, dec_init_s, self.lang_os, max_words, self.model.vocab,
                                                      stop_tokens, self.goal_h)
            self.logprobs.extend(logprobs)

        if outs[-1] == self.corpus.sent2id([SEL])[-1]:
            eos_patch = self.corpus.sent2id([EOS])
            outs += eos_patch
        inpt += outs
        self.read(inpt, self.goal_h, require_speaker=False)
        return outs, self.corpus.id2sent(outs), dec_init_s, dec_last_s

    def update(self, reward):
        self.all_rewards.append(reward)
        # standardize the reward
        rew_r = []
        for i in reward:
            rew_r.append((i - np.mean(reward)) / max(1e-4, np.std(reward)))
        # compute accumulated discounted reward
        rew_g = []
        for r in rew_r:
            rew_g.append(self.model.np2var(np.array([r]), FLOAT).view(1, 1))
        rewards = []
        # print(self.logprobs)
        count = 0
        for p in self.logprobs:
            rewards.insert(0, rew_g[count])
            count = count + 1
            # g = g * self.args.gamma

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.rl_clip)
        self.opt.step()

