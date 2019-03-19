import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

sys.path.append("..")

from utils import bool_flag, initialize_exp
from models import build_model
from trainer import Trainer
from evaluation import Evaluator

VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'

parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
parser.add_argument("--sparse", type=bool_flag, default=False, help="Use SpGAT or GAT")

# data
parser.add_argument("--src_file", type=str, default="./data/processed/zh.processed.vec", help="Source embeddings")
parser.add_argument("--tgt_file", type=str, default="./data/processed/en.processed.vec", help="Target embeddings")
parser.add_argument("--src_nns", type=str, default="./data/processed/zh.processed.adj", help="Source nns")
parser.add_argument("--tgt_nns", type=str, default="./data/processed/en.processed.adj", help="Target nns")
parser.add_argument("--src_lang", type=str, default='zh', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='en', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--enc_dim", type=int, default=512, help="Encoder dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="./data/crosslingual/dictionaries/zh-en.train.txt", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="./data/crosslingual/dictionaries/zh-en.test.txt", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
# GAT parameters
parser.add_argument("--super_steps", type=int, default=5, help="Supervised steps")
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# Decoder
parser.add_argument("--dec_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dec_hid_dim", type=int, default=256, help="Discriminator hidden layer dimensions")
parser.add_argument("--dec_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dec_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
# discriminator
parser.add_argument("--with_dis", type=bool_flag, default=True, help="Discriminator layers")
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=256, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0.5, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=10000, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dec_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")

# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert os.path.isfile(params.src_file)
assert os.path.isfile(params.tgt_file)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

logger = initialize_exp(params)
src_emb, tgt_emb, src_adj, tgt_adj, src_mapping, tgt_mapping, src_decoder, tgt_decoder, discriminator = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, src_adj, tgt_adj, src_mapping, tgt_mapping, src_decoder, tgt_decoder, discriminator, params)
evaluator = Evaluator(trainer)

trainer.load_training_dico(params.dico_train)

logger.info('----> ADVERSARIAL TRAINING <----\n\n')
for n_epoch in range(params.n_epochs):
    logger.info('Starting adversarial training epoch %i...' % n_epoch)
    tic = time.time()
    n_words_proc = 0
    stats = {'DIS_COSTS': [], 'SUPER_COSTS': []}

    for step in range(5):
        trainer.dis_step(stats)

        # if step %
        stats_str = [('DIS_COSTS', 'Discriminator loss'), ('SUPER_COSTS', 'Supervised loss')]
        stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                     for k, v in stats_str if len(stats[k]) > 0]
        # stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
        logger.info(('%06i - ' % n_epoch) + ' - '.join(stats_log))

        # reset
        tic = time.time()
        # n_words_proc = 0
        for k, _ in stats_str:
            del stats[k][:]

    for step in range(int(params.dis_steps)):
        trainer.super_step(stats)
        stats_str = [('DIS_COSTS', 'Discriminator loss'), ('SUPER_COSTS', 'Supervised loss')]
        stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                     for k, v in stats_str if len(stats[k]) > 0]
        # stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
        logger.info(('%06i - ' % n_epoch) + ' - '.join(stats_log))

        # reset
        tic = time.time()
        # n_words_proc = 0
        for k, _ in stats_str:
            del stats[k][:]

    for step in range(int(params.dis_steps)):
        trainer.mapping_step(stats)
        stats_str = [('DIS_COSTS', 'Discriminator loss'), ('SUPER_COSTS', 'Supervised loss')]
        stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                     for k, v in stats_str if len(stats[k]) > 0]
        # stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
        logger.info(('%06i - ' % n_epoch) + ' - '.join(stats_log))

        # reset
        tic = time.time()
        # n_words_proc = 0
        for k, _ in stats_str:
            del stats[k][:]





    # for step in range(params.dis_steps):
    #     trainer.dis_step(stats)
    #
    #     # if step %
    #     stats_str = [('DIS_COSTS', 'Discriminator loss'), ('SUPER_COSTS', 'Supervised loss')]
    #     stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
    #                  for k, v in stats_str if len(stats[k]) > 0]
    #     # stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
    #     logger.info(('%06i - ' % n_epoch) + ' - '.join(stats_log))
    #
    #     # reset
    #     tic = time.time()
    #     # n_words_proc = 0
    #     for k, _ in stats_str:
    #         del stats[k][:]



    stats_str = [('DIS_COSTS', 'Discriminator loss'), ('SUPER_COSTS', 'Supervised loss')]
    stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                 for k, v in stats_str if len(stats[k]) > 0]
    # stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
    logger.info(('%06i - ' % n_epoch) + ' - '.join(stats_log))

    # reset
    tic = time.time()
    # n_words_proc = 0
    for k, _ in stats_str:
        del stats[k][:]


    if n_epoch % 500 == 0:
        stats_str = [('DIS_COSTS', 'Discriminator loss'), ('SUPER_COSTS', 'Supervised loss')]
        stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                     for k, v in stats_str if len(stats[k]) > 0]
        # stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
        logger.info(('%06i - ' % n_epoch) + ' - '.join(stats_log))

        # reset
        tic = time.time()
        # n_words_proc = 0
        for k, _ in stats_str:
            del stats[k][:]

    # embeddings / discriminator evaluation
    to_log = OrderedDict({'n_epoch': n_epoch})
    evaluator.all_eval(to_log)
    evaluator.eval_dis(to_log)


    def default(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    # json.dumps({'value': np.int64(42)}, default=default)

    # JSON log / save best model / end of epoch
    # logger.info("__log__:%s" % json.dumps(to_log, default=default))
    # trainer.save_best(to_log, VALIDATION_METRIC)
    # logger.info('End of epoch %i.\n\n' % n_epoch)

    # update the learning rate (stop if too small)
    trainer.update_lr(to_log, VALIDATION_METRIC)
    if trainer.src_map_optimizer.param_groups[0]['lr'] < params.min_lr:
        logger.info('Learning rate < 1e-6. BREAK.')
        break
    if trainer.tgt_map_optimizer.param_groups[0]['lr'] < params.min_lr:
        logger.info('Learning rate < 1e-6. BREAK.')
        break