from pathlib import Path
from itertools import permutations

def sent_set(data_dir, domains=('restaurants', 'laptops', 'device'), splits=3, modes=('train', 'dev', 'test')):
    res = []
    for domain_a, domain_b in permutations(domains, r=2):
        print('Setting: ' + domain_a + ' to ' + domain_b)
        for split in range(splits):
            for ds in modes:
                ds_file = str(Path(data_dir) / f'{domain_a}_to_{domain_b}_{split + 1}' / f'{ds}.csv')
                print(ds_file)
                ds_text = open(ds_file).read()[len('TOKEN,LABEL,HEAD,HEAD_WORD,DEP_REL,POS,SUB_TOKENS\n'):]
                sents = ds_text.split('_,_,_,_,_,_,_\n_,_,_,_,_,_,_\n')
                for sent in sents:
                    stripped = sent.strip()
                    if stripped:
                        res.append(stripped)
                print()
    return res


orig_data = '/home/daniel_nlp/nlp-architect/nlp_architect/models/libert/cache/libert_benchmark_data/csv'
new_data = '/home/daniel_nlp/nlp-architect/nlp_architect/models/libert/data/csv'

domains = ('device', 'restaurants', 'laptops')
new_sents = sent_set(new_data, domains=domains)
orig_sents = sent_set(orig_data, domains=domains)

i  = 0
sorted_orig = sorted(orig_sents)
sorted_new = sorted(new_sents)

# while True:
#     if sorted_orig[i] != sorted_new[i]:
#         print(sorted_orig[i])
#         print('\n\n')
#         print(sorted_new[i])
#     i += 1

orig_set = set(orig_sents)
new_set = set(new_sents)
diff = orig_set - new_set

print('\n\n\n'.join(diff), '\n')

print('len diff:', len(diff))