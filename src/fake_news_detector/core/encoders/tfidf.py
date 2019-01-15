from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity as cs
from gensim import models

###########################
######### TF-IDF ##########
###########################

def get_topn_relevant_words(cv_model, tf_data, topn=10):
    """get the feature names and tf-idf score of top n items"""
    feature_names =cv_model.get_feature_names()
    sorted_items = sort_coo(tf_data.tocoo())
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results, feature_vals

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def get_cosine_similarity(q1_csc, q2_csc):
    cosine_sim = []
    for i,j in zip(q1_csc, q2_csc):
        sim = cs(i,j)
        cosine_sim.append(sim[0][0])
    return cosine_sim

# From one vocabulary create two different 
# Top N relevant words from TF-IDF algorithm

class Tfidf_models():

    def __init__(self):
        self.cv = None
        self.tfidf_model_real = None
        self.tfidf_model_fake = None



    # Transform text to TF-IDF
    def text2tfidf(self, cv_data, news_type):
        # 2. Translate vec 2 tfidf
        if news_type == 'fake':
            tf_idf_vector = self.tfidf_model_fake.transform(cv_data)
        else:
            tf_idf_vector = self.tfidf_model_real.transform(cv_data)
        return tf_idf_vector


    # Create model for fake and for real
    def create_models(self, cv_fake_news_train, cv_real_news_train):
        self.tfidf_model_fake = TfidfTransformer(use_idf=True).fit(cv_fake_news_train)
        self.tfidf_model_real = TfidfTransformer(use_idf=True).fit(cv_real_news_train)
        return self.tfidf_model_fake, self.tfidf_model_real


    def get_top_words_from_tfidf(self, tfidf_vector, news_type, top_n):
        #tfidf_vector = self.text2tfidf(data_cv, news_type)
        feature_names = self.cv.get_feature_names()
        sorted_tfidf_vector = self.sort_tfdif_by_relevance(tfidf_vector.tocoo())

        #use only topn items from vector
        sorted_tfidf_vector = sorted_tfidf_vector[:top_n]
        score_vals = []
        feature_vals = []
    
        # word index and corresponding tf-idf score
        for idx, score in sorted_tfidf_vector:
            
            #keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
    
        #create a tuples of feature,score
        #results = zip(feature_vals,score_vals)
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        
        return results


    def sort_tfdif_by_relevance(self, tfidf_vector):
        tuples = zip(tfidf_vector.col, tfidf_vector.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def get_relevant_word_lists(self, real_tfidf_vector, fake_tfidf_vector, top_n):
        real_top_words = self.get_top_words_from_tfidf(real_tfidf_vector, 'real', top_n)
        fake_top_words = self.get_top_words_from_tfidf(fake_tfidf_vector, 'fake', top_n)
        return real_top_words, fake_top_words