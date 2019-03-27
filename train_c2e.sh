# !/bin/bash
export CUDA_VISIBLE_DEVICES=1
python supervised.py --cuda True --sparse True --src_file './data/processed/zh.processed.vec' --tgt_file './data/processed/en.processed.vec' --src_nns './data/processed/zh.processed.adj' --tgt_nns './data/processed/en.processed.adj' --src_lang zh --tgt_lang en --dico_train './data/crosslingual/dictionaries/zh-en.all.txt' --dico_eval './data/crosslingual/dictionaries/zh-en.all.txt' --normalize_embeddings center --map_optimizer 'sgd,lr=0.1' --dec_optimizer 'sgd,lr=0.1' --dis_optimizer 'sgd,lr=0.5'
