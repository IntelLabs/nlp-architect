
python3 mark_corpus.py -corpus train.txt -marked_corpus marked_train.txt

python3 examples/np2vec/train.py --workers 30 --size 100 --min_count 10 --window 10 --hs 0 --corpus
 set_expansion_demo/marked_train.txt --np2vec_model_file set_expansion_demo/np2vec --corpus_format txt
 
 - data
 - model wikipedia
 - public service 