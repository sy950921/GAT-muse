import numpy as np
import scipy.sparse as sp
import torch
import io
import re
import os
import sys
import argparse
import re
import inspect
import pickle
import subprocess
import random
from torch import optim
from dictionary import Dictionary
from logging import getLogger
from logger import create_logger

MAIN_DUMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dumped')

logger = getLogger()

# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False


def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization
    if getattr(params, 'seed', -1) >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        if params.cuda:
            torch.cuda.manual_seed(params.seed)

    # dump parameters
    params.exp_path = get_exp_path(params)
    with io.open(os.path.join(params.exp_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # create logger
    logger = create_logger(os.path.join(params.exp_path, 'train.log'))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s' % params.exp_path)
    return logger


def get_exp_path(params):
    """
    Create a directory to store the experiment.
    """
    # create the main dump path if it does not exist
    exp_folder = MAIN_DUMP_PATH if params.exp_path == '' else params.exp_path
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    assert params.exp_name != ''
    exp_folder = os.path.join(exp_folder, params.exp_name)
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    if params.exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            exp_path = os.path.join(exp_folder, exp_id)
            if not os.path.isdir(exp_path):
                break
    else:
        exp_path = os.path.join(exp_folder, params.exp_id)
        assert not os.path.isdir(exp_path), exp_path
    # create the dump folder
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return exp_path


def embedding_read(file_path, full_vocab, params, lang):
    word2id = {}
    vectors = []

    with io.open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        _emb_dim_file = params.emb_dim
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        logger.warning("Word '%s' found twice in embedding file" % word)
                else:
                    if not vect.shape == (_emb_dim_file,):
                        logger.warning("Invalid dimension (%i) for word '%s' in line %i."
                                       % (vect.shape[0], word, i))
                        continue
                    assert vect.shape == (_emb_dim_file,), i
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if params.max_vocab > 0 and len(word2id) >= params.max_vocab and not full_vocab:
                break

    assert len(word2id) == len(vectors)
    logger.info("Loaded %i pre-trained word embeddings." % len(word2id))

    id2word = {v: k for k, v in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def load_nns(nn_file, words, params):
    vectors = []
   # print(words.word2id)
    with io.open(nn_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            word, nns = line.rstrip().split(' ', 1)
            #print(word)
            nns = np.fromstring(nns, sep=' ')
            if word in words.word2id:
                nns = list(nns)
                for nn in nns:
                    vectors.append([i, nn])
            else:
                print('Word %s not in embeddings!' % word)
                exit()

    return np.array(vectors, dtype=np.int32)


def load_src_data(src_file, src_nns, params):
    """Load data"""
    logger.info('Loading data ...')

    src_words, x = embedding_read(src_file, full_vocab=True, params=params, lang=params.src_lang)
    src_edges = load_nns(src_nns, words=src_words, params=params)
    src_adj = sp.coo_matrix((np.ones(src_edges.shape[0]), (src_edges[:, 0], src_edges[:, 1])), shape=(x.shape[0], x.shape[0]), dtype=np.float32)    
    src_adj = torch.FloatTensor(np.array(src_adj.todense()))
    src_adj = normalize_adj(src_adj + torch.eye(src_adj.shape[0]))
    # src_adj = torch.FloatTensor(src_adj)
    src_features = torch.FloatTensor(x)
    return src_words, src_adj, src_features


def load_tgt_data(tgt_file, tgt_nns, params):
    """Load data"""
    logger.info('Loading data ...')

    tgt_words, z = embedding_read(tgt_file, full_vocab=True, params=params, lang=params.tgt_lang)
    tgt_edges = load_nns(tgt_nns, words=tgt_words, params=params)

    tgt_adj = sp.coo_matrix((np.ones(tgt_edges.shape[0]), (tgt_edges[:, 0], tgt_edges[:, 1])),
                            shape=(z.shape[0], z.shape[0]), dtype=np.float32)

    tgt_adj = torch.FloatTensor(np.array(tgt_adj.todense()))
    tgt_adj = normalize_adj(tgt_adj + torch.eye(tgt_adj.shape[0]))
    # tgt_adj = torch.FloatTensor(tgt_adj)
    tgt_features = torch.FloatTensor(z)
    return tgt_words, tgt_adj, tgt_features


# def normalize_adj(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
#     r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
#     return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_adj(mx):
    rowsum = mx.sum(1, keepdim=False)
    r_inv_sqrt = torch.rsqrt(rowsum)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return mx.mm(r_mat_inv_sqrt).transpose(0, 1).mm(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return mean.cpu() if mean is not None else None


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params

def bow(sentences, word_vec, normalize=False):
    """
    Get sentence representations using average bag-of-words.
    """
    embeddings = []
    for sent in sentences:
        sentvec = [word_vec[w] for w in sent if w in word_vec]
        if normalize:
            sentvec = [v / np.linalg.norm(v) for v in sentvec]
        if len(sentvec) == 0:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        embeddings.append(np.mean(sentvec, axis=0))
    return np.vstack(embeddings)


def bow_idf(sentences, word_vec, idf_dict=None):
    """
    Get sentence representations using weigthed IDF bag-of-words.
    """
    embeddings = []
    for sent in sentences:
        sent = set(sent)
        list_words = [w for w in sent if w in word_vec and w in idf_dict]
        if len(list_words) > 0:
            sentvec = [word_vec[w] * idf_dict[w] for w in list_words]
            sentvec = sentvec / np.sum([idf_dict[w] for w in list_words])
        else:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        embeddings.append(np.sum(sentvec, axis=0))
    return np.vstack(embeddings)


def get_idf(europarl, src_lg, tgt_lg, n_idf):
    """
    Compute IDF values.
    """
    idf = {src_lg: {}, tgt_lg: {}}
    k = 0
    for lg in idf:
        start_idx = 200000 + k * n_idf
        end_idx = 200000 + (k + 1) * n_idf
        for sent in europarl[lg][start_idx:end_idx]:
            for word in set(sent):
                idf[lg][word] = idf[lg].get(word, 0) + 1
        n_doc = len(europarl[lg][start_idx:end_idx])
        for word in idf[lg]:
            idf[lg][word] = max(1, np.log10(n_doc / (idf[lg][word])))
        k += 1
    return idf

def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()

def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)

def export_embeddings(src_emb, tgt_emb, params):
    """
    Export embeddings to a text or a PyTorch file.
    """
    assert params.export in ["txt", "pth"]

    # text file
    if params.export == "txt":
        src_path = os.path.join(params.exp_path, 'vectors-%s.txt' % params.src_lang)
        tgt_path = os.path.join(params.exp_path, 'vectors-%s.txt' % params.tgt_lang)
        # source embeddings
        logger.info('Writing source embeddings to %s ...' % src_path)
        with io.open(src_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % src_emb.size())
            for i in range(len(params.src_dico)):
                f.write(u"%s %s\n" % (params.src_dico[i], " ".join('%.5f' % x for x in src_emb[i])))
        # target embeddings
        logger.info('Writing target embeddings to %s ...' % tgt_path)
        with io.open(tgt_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % tgt_emb.size())
            for i in range(len(params.tgt_dico)):
                f.write(u"%s %s\n" % (params.tgt_dico[i], " ".join('%.5f' % x for x in tgt_emb[i])))

    # PyTorch file
    if params.export == "pth":
        src_path = os.path.join(params.exp_path, 'vectors-%s.pth' % params.src_lang)
        tgt_path = os.path.join(params.exp_path, 'vectors-%s.pth' % params.tgt_lang)
        logger.info('Writing source embeddings to %s ...' % src_path)
        torch.save({'dico': params.src_dico, 'vectors': src_emb}, src_path)
        logger.info('Writing target embeddings to %s ...' % tgt_path)
        torch.save({'dico': params.tgt_dico, 'vectors': tgt_emb}, tgt_path)

