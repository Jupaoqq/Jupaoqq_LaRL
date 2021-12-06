import os
import sys
import numpy as np
import torch as th
from torch import nn
from collections import defaultdict
from latent_dialog.enc2dec.base_modules import summary
from latent_dialog.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from datetime import datetime
from latent_dialog.utils import get_detokenize
from latent_dialog.corpora import EOS, PAD
from latent_dialog.data_loaders import DealDataLoaders
from latent_dialog import evaluators
from latent_dialog.record import record, UniquenessSentMetric, UniquenessWordMetric
import logging
import pandas as pd

logger = logging.getLogger()


class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, task, loss):
        if task == "conv":
            for key, val in loss.items():
                # print('key = %s\nval = %s' % (key, val))
                if val is not None and type(val) is not bool:
                    self.losses[key].append(val.item())
        else:
            self.losses['rec'].append(loss)

    def pprint(self, task, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            if task == "conv":
                aver_loss = np.average(loss) if window is None else np.average(loss[-window:])
            else:
                # print(loss)
                total_loss = 0
                total_len = 0
                for i in loss:
                    total_loss = total_loss + i.item()
                    total_len = total_len + 1
                aver_loss = total_loss/total_len
            if 'nll' in key:
                str_losses.append('{} PPL {:.3f}'.format(key, np.exp(aver_loss)))
            else:
                str_losses.append('{} {:.3f}'.format(key, aver_loss))


        if prefix:
            return '{}: {} {}'.format(prefix, name, ' '.join(str_losses))
        else:
            return '{} {}'.format(name, ' '.join(str_losses))

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def avg_loss(self):
        return np.mean(self.backward_losses)



def train(model, train_data, val_data, test_data, config, evaluator, task, gen=None):
    # print("Hello")
    # print(config)
    gen(task, model, val_data, config, evaluator, num_batch=config.preview_batch_num)
    # exit eval model
    # model.train()
    # train_loss.clear()
    # logger.info('\n***** Epoch {}/{} *****'.format(done_epoch, config.max_epoch))
    # sys.stdout.flush()


def validate(task, model, val_data, config, batch_cnt=None, use_py=None):
    model.eval()
    val_data.epoch_init(config, shuffle=False, verbose=False)
    losses = LossManager()
    while True:
        batch = val_data.next_batch()
        if batch is None:
            break
        if use_py is not None:
            loss = model(config, task, batch, mode=TEACH_FORCE, use_py=use_py)
        else:
            loss = model(config, task, batch, mode=TEACH_FORCE)

        losses.add_loss(task, loss)
        losses.add_backward_loss(model.model_sel_loss(task, loss, batch_cnt))

    valid_loss = losses.avg_loss()
    # print('Validation finished at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    logger.info(losses.pprint(task, val_data.name))
    logger.info('Total valid loss = {}'.format(valid_loss))
    sys.stdout.flush()
    return valid_loss


def generate(task, model, data, config, evaluator, num_batch, dest_f=None):
    
    def write(msg):
        if msg is None or msg == '':
            return
        if dest_f is None:
            print(msg)
        else:
            dest_f.write(msg + '\n')

    model.eval()
    de_tknize = get_detokenize()
    data.epoch_init(config, shuffle=False, verbose=False)
    evaluator.initialize()
    logger.info('Generation: {} batches'.format(data.num_batch
                                          if num_batch is None
                                          else num_batch))
    batch_cnt = 0
    print_cnt = 0
    while True:
        print("Batch count")
        print (batch_cnt)
        batch_cnt += 1
        batch = data.next_batch()
        if batch is None or (num_batch is not None and data.ptr > num_batch):
            break

        outputs, labels = model(config, task, batch, mode=GEN, gen_type=config.gen_type)

        # move from GPU to CPU
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
        ctx = batch.get('contexts') # (batch_size, max_ctx_len, max_utt_len)
        ctx_len = batch.get('context_lens') # (batch_size, )

        for b_id in range(pred_labels.shape[0]):
            # TODO attn
            pred_str = get_sent(model.vocab, de_tknize, pred_labels, b_id) 
            true_str = get_sent(model.vocab, de_tknize, true_labels, b_id)
            prev_ctx = ''
            if ctx is not None:
                ctx_str = []
                for t_id in range(ctx_len[b_id]):
                    temp_str = get_sent(model.vocab, de_tknize, ctx[:, t_id, :], b_id, stop_eos=False)
                    # print('temp_str = %s' % (temp_str, ))
                    # print('ctx[:, t_id, :] = %s' % (ctx[:, t_id, :], ))
                    ctx_str.append(temp_str)
                ctx_str = '|'.join(ctx_str)
                prev_ctx = 'Source context: {}'.format(ctx_str)

            evaluator.add_example(true_str, pred_str)

            # if num_batch is None or batch_cnt < 2:
            print_cnt += 1
            write('prev_ctx = %s' % (prev_ctx, ))
            write('True: {}'.format(true_str, ))
            write('Pred: {}'.format(pred_str, ))
            write('='*30)
            # if num_batch is not None and print_cnt > 10:
            #     break

    write(evaluator.get_report())
    write('Generation Done')


def get_sent(vocab, de_tknize, data, b_id, stop_eos=True, stop_pad=True):
    ws = []
    for t_id in range(data.shape[1]):
        w = vocab[data[b_id, t_id]]
        # TODO EOT
        if (stop_eos and w == EOS) or (stop_pad and w == PAD):
            break
        if w != PAD:
            ws.append(w)

    return de_tknize(ws)

















# def train_rec(model, train_data, val_data, test_data, config, evaluator, gen=None):
#     patience = 10
#     valid_loss_threshold = np.inf
#     best_valid_loss = np.inf
#     batch_cnt = 0
#     optimizer = model.get_optimizer(config)
#     done_epoch = 0
#     best_epoch = 0
#     train_loss = LossManager()
#     model.train()
#     logger.info(summary(model, show_weights=False))
#     saved_models = []
#     last_n_model = config.last_n_model if hasattr(config, 'last_n_model') else 5

#     logger.info('***** Training Begins at {} *****'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
#     logger.info('***** Epoch 0/{} *****'.format(config.max_epoch))
#     while True:
#         train_data.epoch_init(config, shuffle=True, verbose=done_epoch==0, fix_batch=config.fix_train_batch)
#         while True:
#             batch = train_data.next_batch()
#             if batch is None:
#                 break
    
#             optimizer.zero_grad()
#             loss = model(batch, mode=TEACH_FORCE)
#             model.backward(loss, batch_cnt)
#             nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
#             optimizer.step()
#             batch_cnt += 1
#             train_loss.add_loss(loss)
    
#             if batch_cnt % config.print_step == 0:
#                 # print('Print step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
#                 logger.info(train_loss.pprint('Train',
#                                         window=config.print_step, 
#                                         prefix='{}/{}-({:.3f})'.format(batch_cnt%config.ckpt_step, config.ckpt_step, model.kl_w)))
#                 sys.stdout.flush()
    
#             if batch_cnt % config.ckpt_step == 0:
#                 logger.info('Checkpoint step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
#                 logger.info('==== Evaluating Model ====')
#                 logger.info(train_loss.pprint('Train'))
#                 done_epoch += 1
#                 logger.info('done epoch {} -> {}'.format(done_epoch-1, done_epoch))

#                 # generation
#                 if gen is not None:
#                     gen(model, val_data, config, evaluator, num_batch=config.preview_batch_num)

#                 # validation
#                 valid_loss = validate(model, val_data, config, batch_cnt)
#                 _ = validate(model, test_data, config, batch_cnt)

#                 # update early stopping stats
#                 if valid_loss < best_valid_loss:
#                     if valid_loss <= valid_loss_threshold * config.improve_threshold:
#                         patience = max(patience, done_epoch*config.patient_increase)
#                         valid_loss_threshold = valid_loss
#                         logger.info('Update patience to {}'.format(patience))
    
#                     if config.save_model:
#                         cur_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
#                         logger.info('!!Model Saved with loss = {},at {}.'.format(valid_loss, cur_time))
#                         th.save(model.state_dict(), os.path.join(config.saved_path, '{}-model'.format(done_epoch)))
#                         best_epoch = done_epoch
#                         saved_models.append(done_epoch)
#                         if len(saved_models) > last_n_model:
#                             remove_model = saved_models[0]
#                             saved_models = saved_models[-last_n_model:]
#                             os.remove(os.path.join(config.saved_path, "{}-model".format(remove_model)))
    
#                     best_valid_loss = valid_loss
    
#                 if done_epoch >= config.max_epoch \
#                         or config.early_stop and patience <= done_epoch:
#                     if done_epoch < config.max_epoch:
#                         logger.info('!!!!! Early stop due to run out of patience !!!!!')
#                     print('Best validation loss = %f' % (best_valid_loss, ))
#                     return best_epoch
    
#                 # exit eval model
#                 model.train()
#                 train_loss.clear()
#                 logger.info('\n***** Epoch {}/{} *****'.format(done_epoch, config.max_epoch))
#                 sys.stdout.flush()


# def validate(model, val_data, config, batch_cnt=None, use_py=None):
#     model.eval()
#     val_data.epoch_init(config, shuffle=False, verbose=False)
#     losses = LossManager()
#     while True:
#         batch = val_data.next_batch()
#         if batch is None:
#             break
#         if use_py is not None:
#             loss = model(batch, mode=TEACH_FORCE, use_py=use_py)
#         else:
#             loss = model(batch, mode=TEACH_FORCE)

#         losses.add_loss(loss)
#         losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

#     valid_loss = losses.avg_loss()
#     # print('Validation finished at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
#     logger.info(losses.pprint(val_data.name))
#     logger.info('Total valid loss = {}'.format(valid_loss))
#     sys.stdout.flush()
#     return valid_loss


# def generate(model, data, config, evaluator, num_batch, dest_f=None):
    
#     def write(msg):
#         if msg is None or msg == '':
#             return
#         if dest_f is None:
#             print(msg)
#         else:
#             dest_f.write(msg + '\n')

#     model.eval()
#     de_tknize = get_detokenize()
#     data.epoch_init(config, shuffle=num_batch is not None, verbose=False)
#     evaluator.initialize()
#     logger.info('Generation: {} batches'.format(data.num_batch
#                                           if num_batch is None
#                                           else num_batch))
#     batch_cnt = 0
#     print_cnt = 0
#     while True:
#         batch_cnt += 1
#         batch = data.next_batch()
#         if batch is None or (num_batch is not None and data.ptr > num_batch):
#             break
#         outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

#         # move from GPU to CPU
#         labels = labels.cpu()
#         pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
#         pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1) # (batch_size, max_dec_len)
#         true_labels = labels.data.numpy() # (batch_size, output_seq_len)

#         # get attention if possible
#         if config.dec_use_attn:
#             pred_attns = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_ATTN_SCORE]]
#             pred_attns = np.array(pred_attns, dtype=float).squeeze(2).swapaxes(0, 1) # (batch_size, max_dec_len, max_ctx_len)
#         else:
#             pred_attns = None
#         # get context
#         ctx = batch.get('contexts') # (batch_size, max_ctx_len, max_utt_len)
#         ctx_len = batch.get('context_lens') # (batch_size, )

#         for b_id in range(pred_labels.shape[0]):
#             # TODO attn
#             pred_str = get_sent(model.vocab, de_tknize, pred_labels, b_id) 
#             true_str = get_sent(model.vocab, de_tknize, true_labels, b_id)
#             prev_ctx = ''
#             if ctx is not None:
#                 ctx_str = []
#                 for t_id in range(ctx_len[b_id]):
#                     temp_str = get_sent(model.vocab, de_tknize, ctx[:, t_id, :], b_id, stop_eos=False)
#                     # print('temp_str = %s' % (temp_str, ))
#                     # print('ctx[:, t_id, :] = %s' % (ctx[:, t_id, :], ))
#                     ctx_str.append(temp_str)
#                 ctx_str = '|'.join(ctx_str)[-200::]
#                 prev_ctx = 'Source context: {}'.format(ctx_str)

#             evaluator.add_example(true_str, pred_str)

#             if num_batch is None or batch_cnt < 2:
#                 print_cnt += 1
#                 write('prev_ctx = %s' % (prev_ctx, ))
#                 write('True: {}'.format(true_str, ))
#                 write('Pred: {}'.format(pred_str, ))
#                 write('='*30)
#                 if num_batch is not None and print_cnt > 10:
#                     break

#     write(evaluator.get_report())
#     write('Generation Done')