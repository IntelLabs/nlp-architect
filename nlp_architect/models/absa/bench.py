from timeit import Timer
from datetime import timedelta

def time(stmt='pass', setup='pass'):
    elapsed = str(timedelta(seconds=Timer(stmt, setup).timeit(1)))
    print(elapsed)
    with open('elapsed.txt', 'w') as f:
        f.write(stmt + ': ' + elapsed)


# time("sleep(1)", "from time import sleep")

time("TrainSentiment(parser='bist').run('/home/daniel_nlp/nlp-architect/nlp_architect/models/absa/ta17.csv')",
        "from train.train import TrainSentiment") # -> 47:18

# time("TrainSentiment(parse=False).run(parsed_data='/home/daniel_nlp/private-nlp-architect/cache/absa/tran/parsed/ta17')",
#         "from train.train import TrainSentiment") # -> 

# time("main(model='en_core_web_sm', n_jobs=4)", "from spacy_mt import main") # -> 00:40

# time("main(model='en_core_web_sm', n_jobs=16)", "from spacy_mt import main") # -> 00:40

# time("main(model='en_core_web_md', n_jobs=4)", "from spacy_mt import main") # -> 3:18

#time("main(model='en_core_web_lg', n_jobs=4)", "from spacy_mt import main") # -> 3:22

# time("main(model='en_core_web_lg', n_jobs=2)", "from spacy_mt import main") # -> 
