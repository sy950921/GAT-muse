# !/bin/bash
SRC_EMB=/home/songyue/translation/GAT-muse/data/monolingual/wiki.en.vec
TGT_EMB=/home/songyue/translation/GAT-muse/data/monolingual/wiki.es.vec
SRC_EMB_OUT=/home/songyue/translation/GAT-muse/data/processed/en_es.en.processed.vec
TGT_EMB_OUT=/home/songyue/translation/GAT-muse/data/processed/en_es.es.processed.vec
SRC_ADJ_OUT=/home/songyue/translation/GAT-muse/data/processed/en_es.en.processed.adj
TGT_ADJ_OUT=/home/songyue/translation/GAT-muse/data/processed/en_es.es.processed.adj

CUDA_VISIBLE_DEVICES=1 python preprocess.py --src_embed $SRC_EMB --tgt_embed $TGT_EMB --src_emb_out $SRC_EMB_OUT --tgt_emb_out $TGT_EMB_OUT --src_adj_out $SRC_ADJ_OUT --tgt_adj_out $TGT_ADJ_OUT --max_vocab 10000
