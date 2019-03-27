import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from utils import load_src_data, load_tgt_data, normalize_embeddings


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from function import ReverseLayerF
from utils import clip_parameters
from utils import get_optimizer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.layer_norm = LayerNorm(nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = self.layer_norm(x)
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)
        self.tanh = torch.nn.Tanh()
        self.layer_norm = LayerNorm(nclass)
        self.emb_norm = EmbedNorm()

    def forward(self, x, adj, alpha):
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.emb_norm(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = self.tanh(x)
        # x = self.layer_norm(x)
        #
        # return self.emb_norm(F.log_softmax(x, dim=1))
        # x = self.emb_norm(x)
        # x = F.log_softmax(x, dim=1)
        # return F.log_softmax(x, dim=1)

        reversed_x = ReverseLayerF.apply(x, alpha)
        return x
        # return self.layer_norm(x)
        # return self.tanh(x)
        # return self.emb_norm(x)
        # return x

class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.enc_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim, (x.dim(), x.size())
        return self.layers(x).view(-1)


class Decoder(nn.Module):

    def __init__(self, params):
        super(Decoder, self).__init__()

        self.emb_dim = params.emb_dim
        self.enc_dim = params.enc_dim
        self.dec_layers = params.dec_layers
        self.dec_hid_dim = params.dec_hid_dim
        self.dec_dropout = params.dec_dropout
        self.dec_input_dropout = params.dec_input_dropout

        layers = [nn.Dropout(self.dec_input_dropout)]
        for i in range(self.dec_layers + 1):
            input_dim = self.enc_dim if i == 0 else self.dec_hid_dim
            output_dim = self.emb_dim if i == self.dec_layers else self.dec_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dec_layers:
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(self.dec_input_dropout))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.enc_dim, (x.dim(), x.size())
        return self.layers(x)


class GATModel(nn.Module):

    def __init__(self, src_encoder, tgt_encoder, src_decoder, tgt_decoder, discriminator):
        super(GATModel, self).__init__()
        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder
        self.src_decoder = src_decoder
        self.tgt_decoder = tgt_decoder
        self.discriminator = discriminator

    def forward(self, src_emb, src_adj, tgt_emb, tgt_adj, dico, params):
        # src_vocab = src_emb.weight.shape[0]
        # tgt_vocab = tgt_emb.weight.shape[0]

        src_vocab = src_emb.shape[0]
        tgt_vocab = tgt_emb.shape[0]

        src_enc = self.src_encoder(src_emb, src_adj, params.alpha)
        tgt_enc = self.tgt_encoder(tgt_emb, tgt_adj, params.alpha)

        x = torch.cat([src_enc, tgt_enc], 0)
        y = torch.FloatTensor(src_vocab + tgt_vocab).zero_()
        y[:src_vocab] = 1 - params.dis_smooth
        y[src_vocab:] = params.dis_smooth
        y = Variable(y.cuda() if params.cuda else y)

        reverse_x = ReverseLayerF.apply(x, params.alpha)
        dis_out = self.discriminator(reverse_x)
        # dis_loss = F.binary_cross_entropy(dis_out, 1-y)

        src_re_emb = self.src_decoder(src_enc)
        tgt_re_emb = self.tgt_decoder(tgt_enc)
        # rebuild_loss = F.mse_loss(src_emb, src_re_emb) + F.mse_loss(tgt_emb, tgt_re_emb)

        A1 = src_enc[dico[:, 0]]
        B1 = tgt_enc[dico[:, 1]]

        # super_loss = F.mse_loss(A1, B1)
        # super_loss += F.mse_loss(src_emb, src_re_emb)
        # super_loss += F.mse_loss(tgt_emb, tgt_re_emb)

        return src_enc, tgt_enc, dis_out, src_re_emb, tgt_re_emb, A1, B1
        


def build_model(params, with_dis):
    """
    Build all components of the model.
    """

    src_dico, src_adj, src_features = load_src_data(params.src_file, params.src_nns, params)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_features), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(src_features)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, tgt_adj, tgt_features = load_tgt_data(params.tgt_file, params.tgt_nns, params)
        params.tgt_dico = tgt_dico
        tgt_emb = nn.Embedding(len(tgt_features), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(tgt_features)
    else:
        tgt_emb = None

    # mapping
    if params.sparse:
        src_mapping = SpGAT(nfeat=params.emb_dim,
                      nhid=params.emb_dim,
                      nclass=params.enc_dim,
                      dropout=params.dropout,
                      nheads=params.nb_heads,
                      alpha=params.alpha)
        tgt_mapping = SpGAT(nfeat=params.emb_dim,
                        nhid=params.emb_dim,
                        nclass=params.enc_dim,
                        dropout=params.dropout,
                        nheads=params.nb_heads,
                        alpha=params.alpha)
    else:
        src_mapping = GAT(nfeat=params.emb_dim,
                    nhid=params.emb_dim,
                    nclass=params.enc_dim,
                    dropout=params.dropout,
                    nheads=params.nb_heads,
                    alpha=params.alpha)
        tgt_mapping = GAT(nfeat=params.emb_dim,
                      nhid=params.emb_dim,
                      nclass=params.enc_dim,
                      dropout=params.dropout,
                      nheads=params.nb_heads,
                      alpha=params.alpha)

    src_decoder = Decoder(params)
    tgt_decoder = Decoder(params)
    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    # cuda
    if params.cuda:
        src_emb.cuda()
        src_adj.cuda()
        src_mapping.cuda()
        src_decoder.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
            tgt_adj.cuda()
            tgt_mapping.cuda()
            tgt_decoder.cuda()
        if with_dis:
            discriminator.cuda()

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    model = GATModel(src_encoder=src_mapping, tgt_encoder=tgt_mapping, src_decoder=src_decoder, tgt_decoder=tgt_decoder, discriminator=discriminator)

    return src_emb, tgt_emb, src_adj, tgt_adj, model, src_dico, tgt_dico


class LayerNorm(nn.Module):
    """Applies Layer Normalization over the last dimension."""

    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.features = features
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.dummy = None
        self.w = None
        self.b = None

    def forward(self, input):
        shape = input.size()

        # In order to force the cudnn path, everything needs to be
        # contiguous. Hence the check here and reallocation below.
        if not input.is_contiguous():
            input = input.contiguous()
        input = input.view(1, -1, shape[-1])

        # Expand w and b buffers if necessary.
        n = input.size(1)
        cur = self.dummy.numel() if self.dummy is not None else 0
        if cur == 0:
            self.dummy = input.data.new(n)
            self.w = input.data.new(n).fill_(1)
            self.b = input.data.new(n).zero_()
        elif n > cur:
            self.dummy.resize_(n)
            self.w.resize_(n)
            self.w[cur:n].fill_(1)
            self.b.resize_(n)
            self.b[cur:n].zero_()
        dummy = self.dummy[:n]
        w = self.w[:n]
        b = self.b[:n]
        output = F.batch_norm(input, dummy, dummy, w, b, True, 0., self.eps)
        return torch.addcmul(self.bias, 1, output.view(*shape), self.gain)


class EmbedNorm(nn.Module):
    """Applies Layer Normalization over the last dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, input):
        shape = input.size()
        if not input.is_contiguous():
            input = input.contiguous()
        input = input.view(1, -1, shape[-1])

        max, _ = torch.max(abs(input), 1)
        x = torch.div(input, max)
        x = x.view(*shape)
        return x
