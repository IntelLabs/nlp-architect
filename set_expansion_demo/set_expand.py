from nlp_architect.models.np2vec import NP2vec


class SetExpand():

    def __init__(self, np2vec_model_fn, mark_char='_', binary=False, word_ngrams=False):
        self.np2vec_model = NP2vec.load(np2vec_model_fn, binary=binary, word_ngrams=word_ngrams)
        self.mark_char = mark_char

    def np2id(self, np):
        return np.replace(' ', self.mark_char) + self.mark_char

    def id2np(self,id):
        return id.replace(self.mark_char, ' ')[:-1]

    def get_vocab(self):
        return set(self.np2vec_model.vocab)

    def expand(self, seed, topn=500):
        seed_ids = list()
        for np in seed:
            id = self.np2id(np)
            if id in self.np2vec_model.vocab:
                seed_ids.append(id)
        if len(seed) > 0:
            res_id = self.np2vec_model.most_similar(seed_ids, topn=topn)
            res = list()
            for r in res_id:
                res.append((self.id2np(r[0]),r[1]))
            return res
        else:
            return None

if __name__ == "__main__":
    se = SetExpand('/Users/jmamou/WorkGoogleDrive/set_expansion_research/experiments/w2v_model/np2vec_concatenate')
    # get vocabulary
    print(se.get_vocab())
    #exp = se.expand(['France','USA','Israel'])
    #print(exp)
