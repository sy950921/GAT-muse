import os
from logging import getLogger

import torch
from torch.autograd import Variable
from torch.nn import functional as F

# from .dico_builder import build_dictionary
from evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary
from utils import clip_parameters
from utils import get_optimizer

logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, src_adj, tgt_adj, src_mapping, tgt_mapping, src_decoder, tgt_decoder, discriminator, params):
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_adj = src_adj
        self.tgt_adj = tgt_adj
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.src_mapping = src_mapping
        self.tgt_mapping = tgt_mapping
        self.src_decoder = src_decoder
        self.tgt_decoder = tgt_decoder
        self.discriminator = discriminator
        self.params = params

        # optimizer
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.src_map_optimizer = optim_fn(src_mapping.parameters(), **optim_params)
            self.tgt_map_optimizer = optim_fn(tgt_mapping.parameters(), **optim_params)
        if hasattr(params, 'dec_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dec_optimizer)
            self.src_decoder_optimizer = optim_fn(src_decoder.parameters(), **optim_params)
            self.tgt_decoder_optimizer = optim_fn(tgt_decoder.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None

        self.best_valid_metrics = -1e12
        self.decrease_lr = False

    def get_dis_xy(self, volatile=True):
        src_vocab = self.src_emb.weight.shape[0]
        tgt_vocab = self.tgt_emb.weight.shape[0]
        src_ids = torch.LongTensor(torch.arange(self.src_emb.weight.shape[0]))
        tgt_ids = torch.LongTensor(torch.arange(self.tgt_emb.weight.shape[0]))
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        with torch.set_grad_enabled(not volatile):
            src_emb = self.src_emb(Variable(src_ids)).detach()
            tgt_emb = self.tgt_emb(Variable(tgt_ids)).detach()
            # src_adj = self.src_adj(Variable(src_ids)).detach()
            # tgt_adj = self.tgt_adj(Variable(tgt_ids)).detach()
            self.src_adj.detach()
            self.tgt_adj.detach()

            new_src_emb = self.src_mapping(src_emb, self.src_adj)
            new_tgt_emb = self.tgt_mapping(tgt_emb, self.tgt_adj)

        # print(src_emb.size())

        x = torch.cat([new_src_emb, new_tgt_emb], 0)
        # print(new_src_emb)
        y = torch.FloatTensor(src_vocab + tgt_vocab).zero_()
        y[:src_vocab] = 1 - self.params.dis_smooth
        y[src_vocab:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y

    def get_new_dis_xy(self, volatile=True):
        src_vocab = self.src_emb.weight.shape[0]
        tgt_vocab = self.tgt_emb.weight.shape[0]
        src_ids = torch.LongTensor(torch.arange(self.src_emb.weight.shape[0]))
        tgt_ids = torch.LongTensor(torch.arange(self.tgt_emb.weight.shape[0]))
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        with torch.set_grad_enabled(not volatile):
            src_emb = self.src_emb(Variable(src_ids)).detach()
            tgt_emb = self.tgt_emb(Variable(tgt_ids)).detach()
            # src_adj = self.src_adj(Variable(src_ids)).detach()
            # tgt_adj = self.tgt_adj(Variable(tgt_ids)).detach()
            self.src_adj.detach()
            self.tgt_adj.detach()

            new_src_emb = self.src_mapping(src_emb, self.src_adj)
            new_tgt_emb = self.tgt_mapping(tgt_emb, self.tgt_adj)

            re_src_emb = self.src_decoder(new_src_emb)
            re_tgt_emb = self.tgt_decoder(new_tgt_emb)

        # print(src_emb.size())

        x = torch.cat([new_src_emb, new_tgt_emb], 0)
        # print(new_src_emb)
        y = torch.FloatTensor(src_vocab + tgt_vocab).zero_()
        y[:src_vocab] = 1 - self.params.dis_smooth
        y[src_vocab:] = self.params.dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y)

        return x, y, src_emb, tgt_emb, re_src_emb, re_tgt_emb

    def dis_step(self, stats):

        self.discriminator.train()
        self.src_mapping.eval()
        self.tgt_mapping.eval()
        
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(Variable(x.data))
        # print(preds)
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.item())

        if (loss != loss).data.any():
            logger.error("NaN detected (discriminator)")
            exit()
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        stats['DIS_COSTS'].append(loss.item())
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):

        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()
        self.src_mapping.train()
        self.tgt_mapping.train()

        x, y, src_emb, tgt_emb, re_src_emb, re_tgt_emb = self.get_new_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        loss = self.params.dis_lambda * loss
        # print(loss.shape)
        # print(src_emb.shape)
        # print(re_src_emb.shape)
        # print(F.mse_loss(src_emb, re_src_emb).shape)
        # print(src_emb)
        # print(re_src_emb)
        # loss = F.mse_loss(src_emb, re_src_emb)
        loss += F.mse_loss(src_emb, re_src_emb)
        loss += F.mse_loss(tgt_emb, re_tgt_emb)

        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        stats['SUPER_COSTS'].append(loss.item())

        self.src_map_optimizer.zero_grad()
        self.tgt_map_optimizer.zero_grad()
        self.src_decoder_optimizer.zero_grad()
        self.tgt_decoder_optimizer.zero_grad()
        loss.backward()
        self.src_decoder_optimizer.step()
        self.tgt_decoder_optimizer.step()
        self.src_map_optimizer.step()
        self.tgt_map_optimizer.step()

    def super_step(self, stats):
        self.discriminator.eval()
        self.src_mapping.train()
        self.tgt_mapping.train()

        volatile = False
        src_ids = torch.LongTensor(torch.arange(self.src_emb.weight.shape[0]))
        tgt_ids = torch.LongTensor(torch.arange(self.tgt_emb.weight.shape[0]))
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        with torch.set_grad_enabled(not volatile):
            src_embeddings = self.src_emb(Variable(src_ids)).detach()
            tgt_embeddings = self.tgt_emb(Variable(tgt_ids)).detach()

            src_emb_new = self.src_mapping(src_embeddings, self.src_adj)
            tgt_emb_new = self.tgt_mapping(tgt_embeddings, self.tgt_adj)
            A1 = src_emb_new[self.dico[:, 0]]
            B1 = tgt_emb_new[self.dico[:, 1]]
            # A2 = A1 / A1.norm(2, 1, keepdim=True).expand_as(A1)
            # B2 = B1 / B1.norm(2, 1, keepdim=True).expand_as(B1)
            A2 = A1
            B2 = B1

            re_src_emb = self.src_decoder(src_emb_new)
            re_tgt_emb = self.tgt_decoder(tgt_emb_new)

        loss = F.mse_loss(A2, B2)
        loss += F.mse_loss(src_embeddings, re_src_emb)
        loss += F.mse_loss(tgt_embeddings, re_tgt_emb)
        if (loss != loss).data.any():
            logger.error("NaN detected (fool discriminator)")
            exit()

        self.src_map_optimizer.zero_grad()
        self.tgt_map_optimizer.zero_grad()
        self.src_decoder_optimizer.zero_grad()
        self.tgt_decoder_optimizer.zero_grad()
        stats['SUPER_COSTS'].append(loss.item())
        loss.backward()
        self.src_decoder_optimizer.step()
        self.tgt_decoder_optimizer.step()
        self.src_map_optimizer.step()
        self.tgt_map_optimizer.step()

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_src_lr = self.src_map_optimizer.param_groups[0]['lr']
        new_src_lr = max(self.params.min_lr, old_src_lr * self.params.lr_decay)
        old_tgt_lr = self.tgt_map_optimizer.param_groups[0]['lr']
        new_tgt_lr = max(self.params.min_lr, old_tgt_lr * self.params.lr_decay)
        if new_src_lr < old_src_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_src_lr, new_src_lr))
            self.src_map_optimizer.param_groups[0]['lr'] = new_src_lr
        if new_tgt_lr < old_tgt_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_tgt_lr, new_tgt_lr))
            self.tgt_map_optimizer.param_groups[0]['lr'] = new_tgt_lr

        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.best_valid_metrics:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            % (to_log[metric], self.best_valid_metrics))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_src_lr = self.src_map_optimizer.param_groups[0]['lr']
                    self.src_map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_src_lr, self.src_map_optimizer.param_groups[0]['lr']))

                    old_tgt_lr = self.tgt_map_optimizer.param_groups[0]['lr']
                    self.tgt_map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                % (old_tgt_lr, self.tgt_map_optimizer.param_groups[0]['lr']))

                self.decrease_lr = True



