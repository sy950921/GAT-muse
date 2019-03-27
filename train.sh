# !/bin/bash
export CUDA_VISIBLE_DEVICES=0 
python supervised.py --cuda True --sparse True --src_file './data/processed/en_es.en.processed.vec' --tgt_file './data/processed/en_es.es.processed.vec' --src_nns './data/processed/en_es.en.processed.adj' --tgt_nns './data/processed/en_es.es.processed.adj' --src_lang en --tgt_lang es --dico_train './data/crosslingual/dictionaries/en-es.0-5000.txt' --dico_eval './data/crosslingual/dictionaries/en-es.txt'
