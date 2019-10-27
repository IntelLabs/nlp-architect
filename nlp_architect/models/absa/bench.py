from timeit import Timer
from datetime import timedelta

def time(stmt='pass', setup='pass'):
    elapsed = str(timedelta(seconds=Timer(stmt, setup).timeit(1)))
    print(elapsed)
    with open('elapsed.txt', 'w') as f:
        f.write(stmt + ': ' + elapsed)


# time("sleep(1)", "from time import sleep")

# time("TrainSentiment(parser='bist').run('/home/daniel_nlp/nlp-architect/nlp_architect/models/absa/ta17.csv')",
#         "from train.train import TrainSentiment") # -> 47:18

# time("TrainSentiment(parser='spacy).run(parsed_data='/home/daniel_nlp/private-nlp-architect/cache/absa/tran/parsed/ta17')",
#         "from train.train import TrainSentiment") # -> 13:38

# time("main(model='en_core_web_sm', n_jobs=4)", "from spacy_mt import main") # -> 00:40

# time("main(model='en_core_web_sm', n_jobs=16)", "from spacy_mt import main") # -> 00:40

# time("main(model='en_core_web_md', n_jobs=4)", "from spacy_mt import main") # -> 3:18

#time("main(model='en_core_web_lg', n_jobs=4)", "from spacy_mt import main") # -> 3:22

# time("main(model='en_core_web_lg', n_jobs=2)", "from spacy_mt import main") # -> 

base = '/data/home/daniel_nlp/nlp-architect/cache/absa/train/lexicons/spacy_ptb_pos_dep_adjust/'
asp_lex = base + 'generated_aspect_lex.csv'
op_lex = base + 'generated_opinion_lex_reranked.csv'

# data = '/data/home/daniel_nlp/nlp-architect/datasets/absa/tripadvisor_co_uk-travel_restaurant_reviews_sample_2000_test.csv'
# time("SentimentInference('" + asp_lex + "', '" + op_lex + "').run_multiple('" + data + "')", "from inference.inference import SentimentInference") # 

# data =  "['This menu is awesome and meals are delicious', 'This starter is awful but vibe is fantastic']"
# time("print(SentimentInference('" + asp_lex + "', '" + op_lex + "').run_multiple(" + data + "))", "from inference.inference import SentimentInference") # 

# data =  'This menu is awesome and meals are delicious'
# time("print(SentimentInference('" + asp_lex + "', '" + op_lex + "').run('" + data + "'))", "from inference.inference import SentimentInference") # 

data =  "/data/home/daniel_nlp/nlp-architect/datasets/absa/tripadvisor_co_uk-travel_restaurant_reviews_sample_2000_test.csv"
# time("print(SentimentInference('" + asp_lex + "', '" + op_lex + "').run_multiple('" + data + "'))", "from inference.inference import SentimentInference") # 


from inference.inference import SentimentInference
print(SentimentInference("/data/home/daniel_nlp/nlp-architect/examples/absa/aspects.csv", "/data/home/daniel_nlp/nlp-architect/examples/absa/opinions.csv").run_multiple(data))