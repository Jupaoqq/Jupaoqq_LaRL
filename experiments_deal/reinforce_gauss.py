import time
import os
import sys
sys.path.append('../')
import json
import torch as th
from latent_dialog.utils import Pack, set_seed
from latent_dialog.corpora import DealCorpus
from latent_dialog import models_deal
from latent_dialog.models_deal import HRED
from latent_dialog.main import Reinforce
from latent_dialog.agent_deal import LatentRlAgent, LstmAgent
from latent_dialog.dialog_deal import Dialog
from latent_dialog.domain import ContextGenerator, ContextGeneratorEval
import warnings
warnings.filterwarnings("ignore")


def main():
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('[START]', start_time, '='*30)

    # RL configuration
    folder = 'gauss_sys_sl'
    epoch_id = '16-rec'
    env = 'gpu'
    sim_epoch_id = '41'
    simulator_folder = 'usr_sl'
    exp_dir = os.path.join('config_log_model', folder, 'rl-' + start_time)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    rl_config = Pack(
        train_path = '../data/negotiate/train.txt',
        val_path = '../data/negotiate/val.txt',
        test_path = '../data/negotiate/test.txt',
        selfplay_path = '../data/negotiate/selfplay.txt',
        selfplay_eval_path = '../data/negotiate/selfplay_eval.txt',
        kg_path='../data/embedding/kg.csv',
        sim_config_path=os.path.join('config_log_model', simulator_folder, 'config.json'),
        sim_model_path=os.path.join('config_log_model', simulator_folder, '{}-model'.format(sim_epoch_id)),
        sv_config_path = os.path.join('config_log_model', folder, 'config.json'), 
        sv_model_path = os.path.join('config_log_model', folder, '{}-model'.format(epoch_id)),
        rl_config_path = os.path.join(exp_dir, 'rl_config.json'),
        rl_model_path = os.path.join(exp_dir, 'rl_model'),
        ppl_best_model_path = os.path.join(exp_dir, 'ppl_best_model'),
        reward_best_model_path = os.path.join(exp_dir, 'reward_best_model'),
        record_path = exp_dir,
        record_freq = 50, 
        use_gpu = env == 'gpu', 
        nepoch = 4,
        nepisode = 500, 
        sv_train_freq = 0, # TODO pay attention to main.py, cuz it is also controlled there
        eval_freq = 0,
        max_words = 100, 
        rl_lr = 0.2, 
        momentum = 0.1, 
        nesterov = True, 
        gamma = 0.95, 
        rl_clip = 1.0,
        ref_text = '../data/negotiate/train.txt',
        domain = 'object_division', 
        max_nego_turn = 50, 
        random_seed = 0,
        use_latent_rl=True
    )



    # save configuration
    with open(rl_config.rl_config_path, 'w') as f:
        json.dump(rl_config, f, indent=4)

    # set random seed
    set_seed(rl_config.random_seed)

    # load previous supervised learning configuration and corpus
    sv_config = Pack(json.load(open(rl_config.sv_config_path)))
    sim_config = Pack(json.load(open(rl_config.sim_config_path)))

    # TODO revise the use_gpu in the config
    sv_config['use_gpu'] = rl_config.use_gpu
    sim_config['use_gpu'] = rl_config.use_gpu
    corpus = DealCorpus(sv_config)

    # load models for two agents
    # TARGET AGENT
    sys_model = models_deal.GaussHRED(corpus, sv_config)
    if sv_config.use_gpu: # TODO gpu -> cpu transfer
        sys_model.cuda()
    sys_model.load_state_dict(th.load(rl_config.sv_model_path, map_location=lambda storage, location: storage))
    # we don't want to use Dropout during RL
    sys_model.eval()
    sys = LatentRlAgent(sys_model, corpus, rl_config, name='System', use_latent_rl=rl_config.use_latent_rl)

    # SIMULATOR we keep usr frozen, i.e. we don't update its parameters
    usr_model = HRED(corpus, sim_config)
    if sim_config.use_gpu:  # TODO gpu -> cpu transfer
        usr_model.cuda()

    usr_model.load_state_dict(th.load(rl_config.sim_model_path, map_location=lambda storage, location: storage))
    usr_model.eval()
    usr_type = LstmAgent
    usr = usr_type(usr_model, corpus, rl_config, name='User')

    user_sent = [line.strip() for line in open('../data/similarity/user.txt')]
    sys_sent = [line.strip() for line in open('../data/similarity/system.txt')]
    # initialize communication dialogue between two agents
    dialog = Dialog([sys, usr], rl_config, user_sent, sys_sent)
    ctx_gen = ContextGenerator(rl_config.selfplay_path)

    # simulation module
    dialog_eval = Dialog([sys, usr], rl_config, user_sent, sys_sent)
    ctx_gen_eval = ContextGeneratorEval(rl_config.selfplay_eval_path)

    
    # start RL
    # reinforce = Reinforce(dialog, ctx_gen, corpus, sv_config, sys_model, usr_model, rl_config, dialog_eval,
    #                       ctx_gen_eval)
    # reinforce.run()

    # save sys model
    th.save(sys_model.state_dict(), rl_config.rl_model_path)

    end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('[END]', end_time, '=' * 30)


if __name__ == '__main__':
    main()