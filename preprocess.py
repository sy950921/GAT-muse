import os
import sys
import io
import numpy as np
import scipy as sp
import argparse
import faiss

def embedding_read(file_path, source, full_vocab, max_vocab):
    word2id = {}
    vectors = []

    res = []
    emb_dim = 0
    with io.open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                _, emb_dim = split
                emb_dim = int(emb_dim)
            else:
                word, vect = line.rstrip().split(' ', 1)
                word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        print("Word '%s' found twice in embedding file" % word)
                else:
                    if not vect.shape == (emb_dim,):
                        print("Invalid dimension (%i) for word '%s' in line %i."
                                       % (vect.shape[0], word, i))
                        continue

                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
                    res.append(line.rstrip().lower())
            if max_vocab > 0 and len(word2id) >= max_vocab:
                break

    assert len(word2id) == len(vectors)
    print("Loaded %i pre-trained word embeddings." % len(word2id))

    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)

    # write into file
    if source:
        if params.src_emb_out:
            outfile = params.src_emb_out
        else:
            outfile = file_path + '.dim%s_vocab%s.content' % (str(emb_dim), str(len(word2id)))
    else:
        if params.tgt_emb_out:
            outfile = params.tgt_emb_out
        else:
            outfile = file_path + '.dim%s_vocab%s.content'%(str(emb_dim), str(len(word2id)))

    if os.path.exists(outfile):
        os.remove(outfile)
    out_f = open(outfile, 'w')
    out_f.write(str(len(word2id)) + ' ' + str(emb_dim) + '\n')
    out_f.write('\n'.join(res))

    return embeddings, id2word

def normalize(matrix):
    norms = sp.sqrt(sp.sum(matrix**2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, sp.newaxis]
    return matrix

def generate_nns(embeddings, id2word, k, outfile):
    if hasattr(faiss, 'StandardGpuResources'):
        # gpu mode
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0
        # index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)

        dim = embeddings.shape[1]
        emb = normalize(embeddings)
        # nbrs = faiss.IndexFlatL2(dim)
        nbrs = faiss.GpuIndexFlatIP(res, dim, config)
        emb_est = np.ascontiguousarray(emb.astype(np.float32))
        nbrs.add(emb_est)
        _, indices = nbrs.search(np.ascontiguousarray(emb.astype(np.float32)), k=k+1)
        indices = indices[:, 1:]

        res = []
        assert len(id2word) == len(indices)
        for i in range(len(id2word)):
            word_vec = [str(int(x)) for x in indices[i]]
            res.append(id2word[i] + ' ' + ' '.join(word_vec))
        # write into file
        if os.path.exists(outfile):
            os.remove(outfile)
        out_f = open(outfile, 'w')
        out_f.write('\n'.join(res))

    return indices

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument("--src_embed", type=str, default="", help="Source embedding file path")
    parser.add_argument("--tgt_embed", type=str, default="", help="Target embedding file path")
    parser.add_argument("--src_emb_out", type=str, default="", help="Source embedding output file path")
    parser.add_argument("--tgt_emb_out", type=str, default="", help="Target embedding output file path")
    parser.add_argument("--src_adj_out", type=str, default="", help="Source adj output file path")
    parser.add_argument("--tgt_adj_out", type=str, default="", help="Target adj output file path")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--k_neighbors", type=int, default=10, help="Find k neighbors for a word")

    params = parser.parse_args()

    src_embeddings, src_id2word = embedding_read(params.src_embed, True, False, params.max_vocab)
    tgt_embeddings, tgt_id2word = embedding_read(params.tgt_embed, False, True, params.max_vocab)

    generate_nns(src_embeddings, src_id2word, params.k_neighbors, params.src_adj_out)
    generate_nns(tgt_embeddings, tgt_id2word, params.k_neighbors, params.tgt_adj_out)
