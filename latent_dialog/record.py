import numpy as np
from latent_dialog.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN, GEN_VALID
from collections import Counter


class UniquenessSentMetric(object):
    """Metric that evaluates the number of unique sentences."""
    def __init__(self):
        self.seen = set()
        self.all_sents = []

    def record(self, sen):
        self.seen.add(' '.join(sen))
        self.all_sents.append(' '.join(sen))

    def value(self):
        return len(self.seen)

    def top_n(self, n):
        return Counter(self.all_sents).most_common(n)


class UniquenessWordMetric(object):
    """Metric that evaluates the number of unique sentences."""
    def __init__(self):
        self.seen = set()

    def record(self, word_list):
        self.seen.update(word_list)

    def value(self):
        return len(self.seen)

def record(n_epsd, model, val_data, sv_config, lm_model, ppl_f, dialog, ctx_gen_eval, rl_f, kg):
    record_ppl_with_lm(n_epsd, model, val_data, sv_config, lm_model, ppl_f)
    record_rl(n_epsd, dialog, ctx_gen_eval, rl_f, kg)

def record_ppl_with_lm(n_epsd, model, data, config, lm_model, ppl_f):
    task = "conv"
    model.eval()
    loss_list = []
    data.epoch_init(config, shuffle=False, verbose=True)
    while True:
        batch = data.next_batch()
        if batch is None:
            break
        for i in range(1):
            loss = model(config, task, batch, mode=TEACH_FORCE, use_py=True)
            loss_list.append(loss.nll.item())

    # USE LM to test generation performance
    data.epoch_init(config, shuffle=False, verbose=False)
    gen_loss_list = []
    # first generate
    while True:
        batch = data.next_batch()
        if batch is None:
            break

        outputs, labels = model(config, task, batch, mode=GEN, gen_type=config.gen_type)
        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1)  # (batch_size, max_dec_len)
        # clean up the pred labels
        clean_pred_labels = np.zeros((pred_labels.shape[0], pred_labels.shape[1]+1))
        clean_pred_labels[:, 0] = model.sys_id
        for b_id in range(pred_labels.shape[0]):
            for t_id in range(pred_labels.shape[1]):
                token = pred_labels[b_id, t_id]
                clean_pred_labels[b_id, t_id + 1] = token
                if token in [model.eos_id] or t_id == pred_labels.shape[1]-1:
                    break

        pred_out_lens = np.sum(np.sign(clean_pred_labels), axis=1)
        max_pred_lens = np.max(pred_out_lens)
        clean_pred_labels = clean_pred_labels[:, 0:int(max_pred_lens)]
        batch['outputs'] = clean_pred_labels
        batch['output_lens'] = pred_out_lens
        loss = lm_model(batch, mode=TEACH_FORCE)
        gen_loss_list.append(loss.nll.item())

    avg_loss = np.average(loss_list)
    avg_ppl = np.exp(avg_loss)
    gen_avg_loss = np.average(gen_loss_list)
    gen_avg_ppl = np.exp(gen_avg_loss)

    ppl_f.write('{}\t{}\t{}\n'.format(n_epsd, avg_ppl, gen_avg_ppl))
    ppl_f.flush()
    model.train()


def record_rl(n_epsd, dialog, ctx_gen, rl_f, kg):
    # conv_list = []
    reward_list = []
    # sys_unique_ = []
    recall_ = []
    dist_ = []
    # sent_metric = UniquenessSentMetric()
    # word_metric = UniquenessWordMetric()

    for ctxs in ctx_gen.ctxs:
        conv, rewards, stats = dialog.run(ctxs, kg, update = False, verbose = False)
        true_reward = rewards[0]
        reward_list.append(stats['system_rew'])
        # sys_unique_.append(stats['system_unique'])
        recall_.append(stats['recall@5'])
        dist_.append(stats['dist-3'])
        # conv_list.append(conv)
        # print("Conv-record")
        # print(conv)
        # for turn in conv:
        #     if turn[0] == 'System':
        #         sent_metric.record(turn[1])
        #         word_metric.record(turn[1])

    # json.dump(conv_list, text_f, indent=4)
    # print(reward_list)
    # avg = []
    # for i in reward_list:
    #     avg.append(np.average(i))
    # aver_reward = np.average(avg)
    # unique_sent_num = sent_metric.value()
    # unique_word_num = word_metric.value()
    # print(sent_metric.top_n(10))
    r = np.array(reward_list).astype(np.float)
    # s = np.array(sys_unique_).astype(np.float)
    ra = np.array(recall_).astype(np.float)
    d = np.array(dist_).astype(np.float)
    aver_reward = np.average(r)
    # sys_unique = np.average(s)
    recall = np.average(ra)
    dist = np.average(d)

    rl_f.write('{}\t{}\t{}\t{}\n'.format(n_epsd, aver_reward, recall, dist))
    rl_f.flush()