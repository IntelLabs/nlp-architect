import json, os
from tqdm import tqdm
import random
from pathlib import Path
from itertools import permutations
import re

DATA_DIR = Path(os.path.realpath(__file__)).parent / 'data'
CROSS_DOMAIN_SETTINGS = ['laptops_to_restaurants', 'restaurants_to_laptops']
IN_DOMAIN_SETTINGS = ['laptops14', 'restaurants14', 'restaurants15', 'restaurants_all', 'service', 'device']
ALL_SETTINGS = CROSS_DOMAIN_SETTINGS + IN_DOMAIN_SETTINGS
NUM_SPLITS = 3
CONLL_DIR = DATA_DIR / 'conll'


def create_dev_sets(domains: list):
    for domain_a, domain_b in tqdm(permutations(domains, r=2)):
        for split in range(1, 4):
            src_dir = CONLL_DIR / (domain_b + '_to_' + domain_a + '_' + str(split))
            try:
                train_text = open(src_dir / 'train.txt', encoding='utf-8').read().strip().split('\n\n')
                dev_len = int(len(train_text) / 3)
                dev_text = train_text[:dev_len]
                target_dir = CONLL_DIR / (domain_a + '_to_' + domain_b + '_' + str(split))
                open(target_dir / 'dev.txt', 'w', encoding='utf-8').write('\n\n'.join(dev_text) + '\n\n')
            except Exception as e:
                raise e

def dai2019_single_to_conll_and_raw(sent_file, tok_file, conll_out: str, raw_out: str, opinion_labels=False):
    """Converts ABSA datasets from Dai (2019) format to CoNLL format.
    Args:
        sentence: Path to textfile sentence desciptors, one json per line.
        token_spans: Path to textfile containing token char ranges
        conll_out: Path for output file.
    """
    sentences = []
    token_spans = []
    aspect_spans = []
    opinions = []

    if isinstance(sent_file, str) and isinstance(tok_file, str):
        with open(sent_file, encoding='utf-8') as sentence_f:
            sent_file = [line for line in sentence_f]
        with open(tok_file, encoding='utf-8') as tok_f:
            tok_file = [line for i, line in enumerate(tok_f) if i % 2 == 1]

    assert len(sent_file) == len(tok_file)

    for json_line in sent_file:
        sent_json = json.loads(json_line)
        sentences.append(sent_json['text'])
        curr_aspects = [term['span'] for term in sent_json['terms']] if 'terms' in sent_json else []
        curr_opinions = sent_json.get('opinions', [])
        opinions.append(curr_opinions)
        aspect_spans.append(curr_aspects)

    for i, line in enumerate(tok_file):
        curr_toks = []
        indices = line.split()
        for j in range(0, len(indices), 2):
            curr_toks.append([int(indices[j]), int(indices[j + 1])])
        token_spans.append(curr_toks)
    assert len(sentences) == len(token_spans) == len(aspect_spans) == len(opinions)

    if raw_out:
        with open(raw_out, 'w', encoding='utf-8') as raw_f:
            raw_f.write('\n'.join(sentences))

    mixed_numeral_pattern = re.compile(' 1/2\tO')
    count_not_found = 0
    conll_sents = []

    for sentence, tok_indices, asp_indices, op_words in tqdm(zip(sentences, token_spans, aspect_spans, opinions)):
        tokens = [sentence[s: e] for s, e in tok_indices]
        tags = ['O' for i in range(len(tokens))]

        if opinion_labels and op_words:
            # if len(op_words) > 2 and [x for x in op_words if ' ' in x]:
            #     print()
            op_words_sorted = sorted([tuple(phrase.lower().split()) for phrase in set(op_words)], key=lambda x: -len(x))
            tok_i = 0
            ops_not_found = set(op_words_sorted)
            while tok_i < len(tokens):
                for op_phrase in op_words_sorted:
                    if len(op_phrase) > 0 and check_match(tokens, tok_i, op_phrase):
                        tags[tok_i] = 'B-OP'
                        for j in range(1, len(op_phrase)):
                            tags[tok_i + j] = 'I-OP'
                        tok_i += len(op_phrase) - 1
                        if op_phrase in ops_not_found:
                            ops_not_found.remove(op_phrase)
                        break
                tok_i += 1

            if ops_not_found:
                count_not_found += 1

        if asp_indices:
            curr_asp = 0
            inside_aspect = False
            for i, (tok_start, tok_end) in enumerate(tok_indices):
                if curr_asp == len(asp_indices):
                    break
                if inside_aspect:
                    tags[i] = 'I-ASP'
                elif tok_start == asp_indices[curr_asp][0]:
                    inside_aspect = True
                    tags[i] = 'B-ASP'
                if tok_end == asp_indices[curr_asp][1]:
                    curr_asp += 1
                    inside_aspect = False

        conll_text = '\n'.join(['\t'.join((_)) for _ in zip(tokens, tags)]) + '\n'
        fixed_conll_text = fix_untoknized_mixed_numerals(conll_text, mixed_numeral_pattern)
        conll_sents.append(fixed_conll_text)

    conll_sents_str = ''.join(conll_sents)

    if conll_out:
        with open(conll_out, 'w', encoding='utf-8') as conll_f:
            conll_f.write(conll_sents_str)

        print(conll_out)
        # print('NOT FOUND: ' + str(count_not_found))
    
    return conll_sents

def fix_untoknized_mixed_numerals(conll_text, pattern):
    fixed_conll_text = pattern.sub('\tO\n1/2\tO', conll_text)
    return fixed_conll_text

def check_match(tokens, i, op_phrase):
    if len(tokens) - i >= len(op_phrase):
        return tuple(tokens[j].lower() for j in range(i, i + len(op_phrase))) == op_phrase
    return False
    
def preprocess_laptops_and_restaurants_dai2019_cross_domain(seed, opinion_labels=True):
    random.seed(seed)
    in_base = str(DATA_DIR / 'Dai2019' / 'semeval')
    out_base = str(DATA_DIR / 'conll') + '/'
    sets = {'restaurants': ('14', '15'), 'laptops': ('14',)}

    for domain, years in sets.items():
        all_domain_sents = get_all_semeval_domain_sents(domain, years, out_base, in_base, opinion_labels)
        
        for split_i in range(NUM_SPLITS):
            split_num = str(split_i + 1)
            lt_to_res_dir = out_base + 'laptops_to_restaurants_' + split_num + '/'
            res_to_lt_dir = out_base + 'restaurants_to_laptops_' + split_num + '/'

            os.makedirs(lt_to_res_dir, exist_ok=True)
            os.makedirs(res_to_lt_dir, exist_ok=True)

            if domain == 'laptops':
                out_test_dir = res_to_lt_dir
                out_train_dir = lt_to_res_dir
            else:
                out_test_dir = lt_to_res_dir
                out_train_dir = res_to_lt_dir
            
            random.shuffle(all_domain_sents)
            split = round(0.75 * len(all_domain_sents))
            train = all_domain_sents[:split]
            test = all_domain_sents[split:]
            
            dai2019_single_to_conll_and_raw([p[0] for p in train], [p[1] for p in train], out_train_dir + 'train.txt', 
                out_train_dir + 'raw_train.txt', opinion_labels)
            dai2019_single_to_conll_and_raw([p[0] for p in test], [p[1] for p in test], out_test_dir + 'test.txt', out_test_dir + 'raw_test.txt',
                opinion_labels)

def count_phrase_in_token_list(phrase, tokens):
    phrase_len = len(phrase)
    spans = []
    for i in range(len(tokens) - phrase_len + 1):
        if tokens[i: i + phrase_len] in (phrase , (phrase[:-1] + [phrase[-1] + 's'])):
            spans.append((i, i + phrase_len))

    return spans

def check_term_list(asp_phrases, op_phrases, tokens, i, stat):
    error = False
    asp_spans = []
    op_spans = []
    for phrase_list in asp_phrases, op_phrases:
        for phrase in phrase_list:
            spans = count_phrase_in_token_list(phrase, tokens)
            count = len(spans)
            if count == 0:
                stat['miss'] += 1
                # print(str(stat['miss']) + ": Missing '" + ' '.join(phrase) + "' in line no. " + str(i + 1))
                # error = True
            if count > 1:
                # print("Multiple '" + ' '.join(phrase) + "' in line no. " + str(i + 1))
                # error = True
                stat['mul'] += 1
            if count !=1:
                stat['err'] += 1
            (asp_spans if phrase_list is asp_phrases else op_spans).extend(spans)

    if error:
        print('Aspects:', asp_phrases)
        print('Opinions:', op_phrases)
        print(' '.join(tokens), '\n')
    return asp_spans, op_spans

def wang_spans_to_conll_sentence(tokens, asp_spans, op_spans):
    labels = ['O' for t in tokens]

    for spans in asp_spans, op_spans:
        for span in spans:
            labels[span[0]] = 'B-ASP' if spans is asp_spans else 'B-OP'
            for i in range(span[0] + 1, span[1]):
                labels[i] = 'I-ASP' if spans is asp_spans else 'I-OP'
    
    return '\n'.join(([t + '\t' + l for t, l in zip(tokens, labels)])) + '\n'

def wang2018_single_to_conll(data_dir, text_op_file, asp_pol_file):
    data_dir = str(data_dir) + '/'
    all_tokens = []
    all_opinions = []
    conll_sents = []
    with open(data_dir + text_op_file, encoding='utf-8') as sent_op_f:
        for line in sent_op_f:
            text_op = line.strip().split('##')
            assert len(text_op) in (1, 2)
            tokens = text_op[0].split()
            opinions = [op_pol.strip(' +-1.,').split() for op_pol in text_op[1].split(' ')] if len(text_op) == 2 else []
            all_tokens.append(tokens)
            all_opinions.append([ops for ops in opinions if ops])
    
    all_aspects = []
    with open(data_dir + asp_pol_file, encoding='utf-8') as asp_pol_f:
        for line in asp_pol_f:
            split_line = line.strip().split(',')
            aspects = []
            if line != 'NIL\n':
                for asp_pol in split_line:
                    aspect = asp_pol.split(':')[0].strip().split()
                    if aspect:
                        aspects.append(aspect)

            all_aspects.append(aspects)
    
    stat = {'miss': 0, 'mul': 0, 'err': 0}
    # with open(data_dir + conll_out, 'w', encoding='utf-8') as out_conll:
    for i, (tokens, aspects, opinions) in enumerate(zip(all_tokens, all_aspects, all_opinions)):
        asp_spans, op_spans = check_term_list(aspects, opinions, tokens, i, stat)
        conll_sentence = wang_spans_to_conll_sentence(tokens, asp_spans, op_spans)
        # out_conll.write(conll_sentence)
        conll_sents.append(conll_sentence)
            
    # print("\nMissing", stat['miss'])
    # print("Multiple", stat['mul'])
    # print('Erroneus lines', stat['err'])
    return conll_sents

def get_all_sentences_of_domain(domain, years):
    semeval_in_base = str(DATA_DIR) + '/Dai2019/semeval'
    all_domain_sents = []
    for year in years:
        out_dir = CONLL_DIR / (domain + year)
        os.makedirs(out_dir, exist_ok=True)
        for ds in 'train', 'test':
            ds_path = semeval_in_base +  year + '/' + domain + '/' + domain + '_' + ds
            sent_file = ds_path + '_sents.json'
            tok_file = ds_path + '_texts_tok_pos.txt'
            with open(sent_file, encoding='utf-8') as sentence_f:
                with open(tok_file, encoding='utf-8') as tok_f:
                    for json_line, tok_line in zip(sentence_f, (line for i, line in enumerate(tok_f) if i % 2 == 1)):
                        all_domain_sents.append((json_line, tok_line))
    return all_domain_sents

def preprocess_wang2018(device_text_op_file, device_asp_pol_file, domain_b, domain_years, seed):
    random.seed(seed)
    out_base = str(DATA_DIR / 'conll') + '/'
    sets = {domain_b: domain_years, 'device': None}

    for domain, years in sets.items():
        if years:
            all_domain_sents = get_all_sentences_of_domain(DATA_DIR, domain, years)
        else:
            all_domain_sents = wang2018_single_to_conll(DATA_DIR, device_text_op_file, device_asp_pol_file)

        for split_i in range(NUM_SPLITS):
            split_num = str(split_i + 1)
            domain_to_device_dir = out_base + domain_b + '_to_device_' + split_num + '/'
            device_to_domain_dir = out_base + 'device_to_' + domain_b + '_' + split_num + '/'
            os.makedirs(device_to_domain_dir, exist_ok=True)
            os.makedirs(domain_to_device_dir, exist_ok=True)

            if domain == 'device':
                out_test_dir = domain_to_device_dir
                out_train_dir = device_to_domain_dir
            else:
                out_test_dir = device_to_domain_dir
                out_train_dir = domain_to_device_dir
            
            random.shuffle(all_domain_sents)
            split = round(0.75 * len(all_domain_sents))
            train = all_domain_sents[:split]
            test = all_domain_sents[split:]
            
            if years:
                dai2019_single_to_conll_and_raw([p[0] for p in train], [p[1] for p in train], out_train_dir + 'train.txt', 
                    out_train_dir + 'raw_train.txt', opinion_labels=True)
                dai2019_single_to_conll_and_raw([p[0] for p in test], [p[1] for p in test], out_test_dir + 'test.txt', out_test_dir + 'raw_test.txt',
                    opinion_labels=True)           
            else:
                with open(out_train_dir + 'train.txt', 'w', encoding='utf-8') as f:
                    f.write('\n'.join(train) + '\n')
                with open(out_test_dir + 'test.txt', 'w', encoding='utf-8') as f:
                    f.write('\n'.join(test) + '\n')

def get_all_semeval_domain_sents(domain, years, out_base, in_base, opinion_labels):
    all_domain_sents = []
    for year in years:
        out_dir = out_base + domain + year + '/'
        os.makedirs(out_dir, exist_ok=True)
        for ds in 'train', 'test':
            ds_path = in_base +  year + '/' + domain + '/' + domain + '_' + ds
            sent_file = ds_path + '_sents.json'
            tok_file = ds_path + '_texts_tok_pos.txt'
            dai2019_single_to_conll_and_raw(sent_file, tok_file, out_dir + ds + '.txt', 
                out_dir + 'raw_' + ds + '.txt', opinion_labels)

            with open(sent_file, encoding='utf-8') as sentence_f:
                with open(tok_file, encoding='utf-8') as tok_f:
                    for json_line, tok_line in zip(sentence_f, (line for i, line in enumerate(tok_f) if i % 2 == 1)):
                        all_domain_sents.append((json_line, tok_line))
    return all_domain_sents

def dai2019_domain_to_conll(domain, years):
    domain_sents = get_all_sentences_of_domain(domain, years)
    sent_file = [tup[0] for tup in domain_sents]
    tok_file = [tup[1] for tup in domain_sents]
    conll_sents = dai2019_single_to_conll_and_raw(
        sent_file, tok_file, conll_out=False, raw_out=False, opinion_labels=True)
    return conll_sents

def preprocess_2_domains_to_1(seed):
    random.seed(seed)
    device_text_op_file, device_asp_pol_file = 'Wang2018/addsenti_device', 'Wang2018/aspect_op_device'

    domain_sets = {'restaurants': ('14', '15'), 'laptops': ('14',), 'device': None}
    dom_sentences = {}
    for domain, years in domain_sets.items():
        dom_sentences[domain] = dai2019_domain_to_conll(domain, years) if years \
            else wang2018_single_to_conll(DATA_DIR, device_text_op_file, device_asp_pol_file)

    dev_test_sets = {}
    for domain, sentences in dom_sentences.items():  
        random.shuffle(sentences)
        mid_point = len(sentences) // 2
        dev_test_sets[domain] = sentences[:mid_point], sentences[mid_point:]

    domains = list(domain_sets.keys())
    for i in range(len(domains)):
        target_domain = domains[i]
        src_domain_1 = domains[(i + 1) % 3]
        src_domain_2 = domains[(i + 2) % 3]
        data_dir = CONLL_DIR / f"{src_domain_1}_{src_domain_2}_to_{target_domain}"
        os.makedirs(data_dir, exist_ok=True)

        data = {'train': dom_sentences[src_domain_1] + dom_sentences[src_domain_2],
                'test': dev_test_sets[target_domain][0],
                'dev': dev_test_sets[target_domain][1]}

        for set_name, set_data in data.items():
            with open(data_dir / f'{set_name}.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(set_data) + '\n')

def preprocess_devices_wang2018_cross_domain(seed):
    device_text_op_file, device_asp_pol_file = 'Wang2018/addsenti_device', 'Wang2018/aspect_op_device'
    for domain, years in [('restaurants', ('14', '15')), ('laptops', ('14',))]:
        preprocess_wang2018(device_text_op_file, device_asp_pol_file, domain, years, seed)


def prepare_all_datasets(seed):
    preprocess_laptops_and_restaurants_dai2019_cross_domain(seed=seed)
    preprocess_devices_wang2018_cross_domain(seed=seed)
    create_dev_sets(domains=['laptops', 'restaurants', 'device'])

if __name__ == "__main__":
    # prepare_all_datasets(seed=16)
    preprocess_2_domains_to_1(seed=16)