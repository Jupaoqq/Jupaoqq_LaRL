import torch as th
import torch.nn as nn
from torch.autograd import Variable
from latent_dialog.base_models import BaseModel
from latent_dialog.corpora import SYS, EOS, PAD
from latent_dialog.utils import INT, FLOAT, LONG, Pack
from latent_dialog.enc2dec.encoders import EncoderRNN, RnnUttEncoder
from latent_dialog.enc2dec.decoders import DecoderRNN
from latent_dialog.nn_lib import IdentityConnector, Bi2UniConnector
from latent_dialog.enc2dec.decoders import DecoderRNN, GEN, GEN_VALID, TEACH_FORCE
from latent_dialog.criterions import NLLEntropy, NLLEntropy4CLF, CombinedNLLEntropy4CLF
import latent_dialog.utils as utils
import latent_dialog.nn_lib as nn_lib
import latent_dialog.criterions as criterions
import numpy as np
import pandas as pd
from latent_dialog.utils import get_detokenize
from latent_dialog.recommendation import softmax_func, movieID_to_embedding
from latent_dialog.main import get_sent

# def get_sent(vocab, de_tknize, data, b_id, stop_eos=True, stop_pad=True):
#     ws = []
#     for t_id in range(data.shape[1]):
#         w = vocab[data[b_id, t_id]]
#         # TODO EOT
#         if (stop_eos and w == EOS) or (stop_pad and w == PAD):
#             break
#         if w != PAD:
#             ws.append(w)

#     return de_tknize(ws)

class HRED(BaseModel):
    def __init__(self, corpus, config):
        super(HRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.person = config.person
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=1,
                                         goal_nhid=config.goal_nhid,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0,
                                      rnn_cell=config.ctx_rnn_cell,
                                      # input_size=self.utt_encoder.output_size+config.goal_nhid, 
                                      input_size=self.utt_encoder.output_size,
                                      hidden_size=config.ctx_cell_size,
                                      num_layers=config.num_layers,
                                      output_dropout_p=config.dropout,
                                      bidirectional=config.bi_ctx_cell,
                                      variable_lengths=False)

        # TODO connector
        if config.bi_ctx_cell:
            self.connector = Bi2UniConnector(rnn_cell=config.ctx_rnn_cell,
                                             num_layer=1,
                                             hidden_size=config.ctx_cell_size,
                                             output_size=config.dec_cell_size)
        else:
            self.connector = IdentityConnector()

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size + config.goal_nhid,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.ctx_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.sys_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        clf = False
        if not clf:
            ctx_lens = data_feed['context_lens']  # (batch_size, )cc
            ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
            ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)  # (batch_size, max_ctx_len)
            out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
            # goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
            batch_size = len(ctx_lens)
            # if self.person == "user":
            #   goals_h = th.zeros((batch_size, 128), device='cuda')
            # elif self.person == "system":
            #   goals_h = self.np2var(data_feed['usr_mention'], FLOAT)
            goals_h = th.zeros((batch_size, 128), device='cuda')
            # encode goal info
            enc_inputs, _, _ = self.utt_encoder(ctx_utts, feats=ctx_confs,
                                                goals=goals_h)  # (batch_size, max_ctx_len, num_directions*utt_cell_size)

            # enc_outs: (batch_size, max_ctx_len, ctx_cell_size)
            # enc_last: tuple, (h_n, c_n)
            # h_n: (num_layers*num_directions, batch_size, ctx_cell_size)
            # c_n: (num_layers*num_directions, batch_size, ctx_cell_size)
            enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)

            # get decoder inputs
            dec_inputs = out_utts[:, :-1]
            labels = out_utts[:, 1:].contiguous()

            # pack attention context
            if self.config.dec_use_attn:
                attn_context = enc_outs
            else:
                attn_context = None

            # create decoder initial states
            dec_init_state = self.connector(enc_last)

            # decode
            dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                                   dec_inputs=dec_inputs,
                                                                   # (batch_size, response_size-1)
                                                                   dec_init_state=dec_init_state,  # tuple: (h, c)
                                                                   attn_context=attn_context,
                                                                   # (batch_size, max_ctx_len, ctx_cell_size)
                                                                   mode=mode,
                                                                   gen_type=gen_type,
                                                                   beam_size=self.config.beam_size,
                                                                   goal_hid=goals_h)  # (batch_size, goal_nhid)

            
            # print(ret_dict)
            # print(labels)

            if mode == GEN:
                return ret_dict, labels
            if return_latent:
                return Pack(nll=self.nll(dec_outputs, labels),
                            latent_action=dec_init_state)
            else:
                return Pack(nll=self.nll(dec_outputs, labels))




class GaussHRED(BaseModel):
    def __init__(self, corpus, config):
        super(GaussHRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.person = config.person
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        
        self.simple_posterior = config.simple_posterior

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=config.goal_nhid,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0,
                                      rnn_cell=config.ctx_rnn_cell,
                                      # input_size=self.utt_encoder.output_size+config.goal_nhid,
                                      input_size=self.utt_encoder.output_size,
                                      hidden_size=config.ctx_cell_size,
                                      num_layers=config.num_layers,
                                      output_dropout_p=config.dropout,
                                      bidirectional=config.bi_ctx_cell,
                                      variable_lengths=False)
        # mu and logvar projector
        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size, config.y_size, is_lstm=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(config.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size+self.ctx_encoder.output_size, config.y_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size + config.goal_nhid,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.ctx_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.sys_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.gauss_kl = criterions.NormKLLoss(unit_average=True)
        self.zero = utils.cast_type(th.zeros(1), FLOAT, self.use_gpu)
        self.w_matrix = nn.Parameter(th.randn(256, 128, device='cuda'))
        self.kg = pd.read_csv(self.config.kg_path, header = None)
        self.kg = self.kg.astype({128: int})

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

    def z2dec(self, last_h, requires_grad):
        p_mu, p_logvar = self.c2z(last_h)
        if requires_grad:
            sample_z = self.gauss_connector(p_mu, p_logvar)
            joint_logpz = None
        else:
            sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
            logprob_sample_z = self.gaussian_logprob(p_mu, p_logvar, sample_z)
            joint_logpz = th.sum(logprob_sample_z.squeeze(0), dim=1)

        dec_init_state = self.z_embedding(sample_z)
        attn_context = None

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        return dec_init_state, attn_context, joint_logpz



    def forward(self, config, task, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        


        ctx_lens = data_feed['context_lens']  # (batch_size, )cc
        ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        usr_mention = data_feed['usr_mention']
        sys_mention = data_feed['sys_mention']
        movies = data_feed['movies']
        movies_batch = []
        for mov in movies:
          movie = dict()
          for i in range(len(mov)//4):
              movie[mov[i*4]] = {}
              movie[mov[i*4]]['suggested'] = mov[i*4+1]
              movie[mov[i*4]]['seen'] = mov[i*4+2]
              movie[mov[i*4]]['liked'] = mov[i*4+3]
          movies_batch.append(movie)
        # print(movies_batch)
        # print("Hello")
        # print(len(data_feed['sys_mention']))  # (batch_size, max_out_len)
        # goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
        batch_size = len(ctx_lens)
        # if self.person == "user":
        #   goals_h = th.zeros((batch_size, 128), device='cuda')
        # elif self.person == "system":
        #   goals_h = self.np2var(data_feed['usr_mention'], FLOAT)
        goals_h = th.zeros((batch_size, 128), device='cuda')

        enc_inputs, _, _ = self.utt_encoder(ctx_utts, goals=goals_h)
        # (batch_size, max_ctx_len, num_directions*utt_cell_size)

        # enc_outs: (batch_size, max_ctx_len, ctx_cell_size)
        # enc_last: tuple, (h_n, c_n)
        # h_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        # c_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        # print(ctx_utts[0])
        # print(out_utts[0])
        # print(usr_mention[0])
        # print(sys_mention[0])
        # print(movies[0])
        # print(dec_inputs[0])
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1),  goals=goals_h)
            q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1).unsqueeze(0)], dim=2))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z)
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        mode2 = mode
        if task == "rec":
          mode2 = GEN
        # decode
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode2,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size,
                                                               goal_hid=goals_h)  # (batch_size, goal_nhid)

        # print("task")
        # print(dec_outputs.shape)
        # print(labels.shape)
        if task == "conv":
          if mode == GEN:
              ret_dict['sample_z'] = sample_z
              return ret_dict, labels
          else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result


        if task == "rec":
          # print(mode)
          # print(ret_dict[DecoderRNN.KEY_SEQUENCE])
          if len(ret_dict[DecoderRNN.KEY_SEQUENCE]) > 0:
            de_tknize = get_detokenize()
            ret_dict['sample_z'] = sample_z
            outputs = ret_dict
            labels = labels.cpu()
            pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]

            pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1) # (batch_size, max_dec_len)
            true_labels = labels.data.numpy() # (batch_size, output_seq_len)

            # get attention if possible
            if config.dec_use_attn:
                pred_attns = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_ATTN_SCORE]]
                pred_attns = np.array(pred_attns, dtype=float).squeeze(2).swapaxes(0, 1) # (batch_size, max_dec_len, max_ctx_len)
            else:
                pred_attns = None
            # get context
            ctx = data_feed.get('contexts') # (batch_size, max_ctx_len, max_utt_len)
            ctx_len = data_feed.get('context_lens') # (batch_size, )
            # print(pred_labels.shape)
            loss_total = 0
            loss_len = 0

            for b_id in range(pred_labels.shape[0]):
                # TODO attn
                pred_str = get_sent(self.vocab, de_tknize, pred_labels, b_id) 
                true_str = get_sent(self.vocab, de_tknize, true_labels, b_id)

                z = dec_init_state
                # if '[ITEM]' in true_str and '[ITEM]' in pred_str:
                # print(movies_batch[b_id])
                t = z[0][b_id]
                t = t[None, None, :]

                embed, m_id, loss = softmax_func(t, movies_batch[b_id], self.kg, self.w_matrix, True)
                # print(m_id)
                if loss != -1:
                  loss_total = loss_total + loss
                  loss_len = loss_len + 1
            avg_loss = loss_total / loss_len
            # print(loss_total)
            # print(avg_loss)
            if mode == GEN:
              ret_dict['sample_z'] = sample_z
              return ret_dict, labels
            else:
              return avg_loss
          else:
            print("error")
            # if "[ITEM]" in pred_str:
            #   embed, m_id = self.softmax_func(z, profile, kg)


            # if mode == GEN:
            #   ret_dict['sample_z'] = sample_z
            #   return dec_init_state, self.w_matrix, ret_dict, mention, movies
            # else: 
            #   return

          return

        


class CatHRED(BaseModel):
    def __init__(self, corpus, config):
        super(CatHRED, self).__init__(config)

        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.person = config.person
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.simple_posterior = config.simple_posterior

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=config.goal_nhid,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0,
                                      rnn_cell=config.ctx_rnn_cell,
                                      # input_size=self.utt_encoder.output_size+config.goal_nhid,
                                      input_size=self.utt_encoder.output_size,
                                      hidden_size=config.ctx_cell_size,
                                      num_layers=config.num_layers,
                                      output_dropout_p=config.dropout,
                                      bidirectional=config.bi_ctx_cell,
                                      variable_lengths=False)
        # mu and logvar projector
        self.c2z = nn_lib.Hidden2Discrete(self.ctx_encoder.output_size, config.y_size, config.k_size,
                                          is_lstm=config.ctx_rnn_cell == 'lstm')

        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Discrete(self.ctx_encoder.output_size + self.utt_encoder.output_size,
                                               config.y_size, config.k_size, is_lstm=False)

        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        self.z_embedding = nn.Linear(config.y_size * config.k_size, config.dec_cell_size, bias=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size + config.goal_nhid,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.ctx_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.sys_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.entropy_loss = criterions.Entropy()

        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss -= self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def z2dec(self, last_h, requires_grad):
        logits, log_qy = self.c2z(last_h)

        if requires_grad:
            sample_y = self.gumbel_connector(logits)
            logprob_z = None
        else:
            idx = th.multinomial(th.exp(log_qy), 1).detach()
            logprob_z = th.sum(log_qy.gather(1, idx))
            sample_y = utils.cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_y.scatter_(1, idx, 1.0)

        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.config.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.config.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            attn_context = None
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))

        return dec_init_state, attn_context, logprob_z

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )cc
        ctx_utts = self.np2var(data_feed['contexts'], LONG)  # (batch_size, max_ctx_len, max_utt_len)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        # goals = self.np2var(data_feed['goals'], LONG)  # (batch_size, goal_len)
        batch_size = len(ctx_lens)
        # if self.person == "user":
        #   goals_h = th.zeros((batch_size, 128), device='cuda')
        # elif self.person == "system":
        #   goals_h = self.np2var(data_feed['usr_mention'], FLOAT)
        goals_h = th.zeros((batch_size, 128), device='cuda')

        # # encode goal info
        enc_inputs, _, _ = self.utt_encoder(ctx_utts, goals=goals_h)
        # (batch_size, max_ctx_len, num_directions*utt_cell_size)

        # enc_outs: (batch_size, max_ctx_len, ctx_cell_size)
        # enc_last: tuple, (h_n, c_n)
        # h_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        # c_n: (num_layers*num_directions, batch_size, ctx_cell_size)
        enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1), goals=goals_h)
            logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1).unsqueeze(0)], dim=2))
            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_y = self.gumbel_connector(logits_py)
            else:
                sample_y = self.gumbel_connector(logits_qy)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.config.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.config.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            attn_context = None
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))

        # decode
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size,
                                                               goal_hid=goals_h)  # (batch_size, goal_nhid)


        if mode == GEN:
            return ret_dict, labels
        else:
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            pi_h = self.entropy_loss(log_qy, unit_average=True)
            results = Pack(nll=self.nll(dec_outputs, labels), mi=mi, pi_kl=pi_kl, pi_h=pi_h)

            if return_latent:
                results['latent_action'] = dec_init_state

            return results
