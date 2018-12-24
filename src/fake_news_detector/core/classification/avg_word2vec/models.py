from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
import gensim
from sklearn.model_selection import train_test_split
import multiprocessing
import random


## TOPIC MODELING

# def get_model(var, corpus, n_topics, dictionary):
#     switcher = {
#         'LSI': LsiModel(corpus=corpus, num_topics=n_topics, id2word=dictionary),
#         'HDP': HdpModel(corpus=corpus, id2word=dictionary),
#         'LDA':  LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary)
#     }
#     return switcher.get(var, -1)

cores = multiprocessing.cpu_count()

class AveWord2VecModels():

    def __init__(self, data, tagged_documents):
        self.documents = tagged_documents
        self.labels = data['label']
        self.ids = data['id']
        self.models = [
                # PV-DBOW 
                Word2Vec.load_word2vec_format("/data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz", binary=True)
        ]

    def build_models(self):
        self.models[0].build_vocab(self.documents)
        print(str(self.models[0]))
        self.models[1].build_vocab(self.documents)
        print(str(self.models[1]))
            

    """ TRAINING """

    def train_models(self):
        for model in self.models:
            model.init_sims(replace=True)

    def retrain_vector(self, vec):
        vectors_test = []
        for model in self.models:
            v = model.infer_vector(vec)
            vectors_test.append(v)
        return vectors_test
        


    """ TEST """
    def test_document(self, test_corpus):
        doc_id = random.randint(0, len(test_corpus) - 1)
        for model in self.models:
            inferred_vector = model.infer_vector(test_corpus[doc_id])
            sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

            # Compare and print the most/median/least similar documents from the train corpus
            print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
            print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
            for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
                print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(self.corpus[sims[index][0]].words)))
    

    """ GETTERS """

    def get_similar_from_text(self,text):
        results = []
        for model in self.models:
            res = model.docvecs.most_similar(positive=text, topn=10)
            results.append(res)
        return results
    
    def get_similar_from_vec(self,vec):
        results = []
        for model in self.models:
            res = model.docvecs.most_similar(vec, topn=10)
            results.append(res)
        return results

    

# For evaluate each classificator
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim_news_classification.ipynb