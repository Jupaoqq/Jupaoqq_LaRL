import time
import os
import sys
sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import torch as th
import logging
from latent_dialog.utils import Pack, prepare_dirs_loggers, set_seed
from latent_dialog.corpora import DealCorpus
from latent_dialog.data_loaders import DealDataLoaders
from latent_dialog.evaluators import BleuEvaluator
from latent_dialog.models_deal import HRED
from latent_dialog.main_bleu import train, validate, generate
import latent_dialog.domain as domain


stats_path = 'config_log_model'
if not os.path.exists(stats_path):
    os.mkdir(stats_path)
start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
print('[START]', start_time, '='*30)

domain_name = 'object_division'
domain_info = domain.get_domain(domain_name)

#edit mode, best_conv_model, pre_trained folder
mode = "rec_only"
best_conv_model = 42

config = Pack(
    person = sys.argv[1],
    random_seed = 10,
    train_path = '../data/negotiate/train.txt', 
    val_path = '../data/negotiate/val.txt', 
    test_path = '../data/negotiate/test.txt', 
    kg_path='../data/embedding/kg.csv',
    last_n_model = 4, 
    max_utt_len = 20,
    backward_size = 14, 
    batch_size = 32, 
    use_gpu = True,
    op = 'adam', 
    init_lr = 0.001, 
    l2_norm = 0.00001, 
    momentum = 0.0,
    grad_clip=10.0,
    dropout = 0.3, 
    max_epoch = 50, 
    embed_size = 256, 
    num_layers = 1, 
    utt_rnn_cell = 'gru', 
    utt_cell_size = 128, 
    bi_utt_cell = True, 
    enc_use_attn = False, 
    ctx_rnn_cell = 'gru',
    ctx_cell_size = 256,
    bi_ctx_cell = False,
    dec_use_attn = True,
    dec_rnn_cell = 'gru', # must be same as ctx_cell_size due to the passed initial state
    dec_cell_size = 256, # must be same as ctx_cell_size due to the passed initial state
    dec_attn_mode = 'cat', 
    #
    beam_size = 20,
    fix_train_batch = False, 
    avg_type = 'real_word',
    print_step = 100,
    ckpt_step = 400,
    improve_threshold = 0.996, 
    patient_increase = 2.0, 
    save_model = True, 
    early_stop = False, 
    gen_type = 'greedy', 
    preview_batch_num = 50, 
    max_dec_len = 40, 
    k = domain_info.input_length(), 
    goal_embed_size = 128, 
    goal_nhid = 128, 
    init_range = 0.1,
    pretrain_folder ='sys_sl_word',
    forward_only = False
)

set_seed(config.random_seed)

if mode == "rec_only": 
    saved_path = os.path.join(stats_path, config.pretrain_folder)
    # config = Pack(json.load(open(os.path.join(saved_path, 'config.json'))))
    config['forward_only'] = True
else:
    saved_path = os.path.join(stats_path, start_time+'-'+os.path.basename(__file__).split('.')[0])
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

config.saved_path = saved_path

prepare_dirs_loggers(config)
logger = logging.getLogger()
logger.info('[START]\n{}\n{}'.format(start_time, '='*30))

# save configuration
with open(os.path.join(saved_path, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4) # sort_keys=True

corpus = DealCorpus(config)
train_dial, val_dial, test_dial = corpus.get_corpus()

train_data = DealDataLoaders('Train', train_dial, config)
val_data = DealDataLoaders('Val', val_dial, config)
test_data = DealDataLoaders('Test', test_dial, config)

evaluator = BleuEvaluator('Deal')

model = HRED(corpus, config)

if config.use_gpu:
    model.cuda()

best_epoch = None

if config.person != "user" and mode != "conv_only":
    config.max_epoch = 20
    task = "rec"
    if mode == "both":
        best_conv_model = best_epoch
    model.load_state_dict(th.load(os.path.join(saved_path, '{}-model'.format(best_conv_model))))
    best_epoch2 = None
 
    try:
        best_epoch2 = train(model, train_data, val_data, test_data, config, evaluator, task, gen=generate)
    except KeyboardInterrupt:
        print('Training stopped by keyboard.')


end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
logger.info('[END]'+ end_time+ '='*30)