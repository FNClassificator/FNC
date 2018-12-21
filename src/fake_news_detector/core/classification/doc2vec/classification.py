
from gensim.models.doc2vec import TaggedDocument
from src.fake_news_detector.core.classification.doc2vec_models import Doc2VecModels
import gensim
"""
Build taggeds documents to build Doc2vec model

@ data: 
    - corpus: Tokenized words
    - id: str to identify each document
"""
def get_corpus(data):
    documents = []
    for _, row in data.iterrows():
        tagged_doc = gensim.models.doc2vec.TaggedDocument(row['corpus'], [row['id']])
        documents.append(tagged_doc)
    return documents

"""
Build taggeds documents to build Doc2vec model

@ data: 
    - corpus: Tokenized words
    - id: str to identify each document
    - label: label to classify
"""
def generate_doc2vec_model(data):
    tagged_documents = get_corpus(data)
    # 1. Create
    models = Doc2VecModels(data, tagged_documents)
    # 2. Build vocabulary
    models.build_models()
    # 3. Train model
    models.train_models()
    return models

"""
Build taggeds documents to build Doc2vec model

@ data: 
    - corpus: Tokenized words
    - id: str to identify each document
    - label: label to classify
"""
def get_similarty_doc2vec(models, data):
    # 1. Build model
    models = generate_doc2vec_model(data)
    # 2. Test similarity
    result_m1_max = []
    result_m1_mean = []
    result_m2_max = []
    result_m2_mean = []
    # Check the mosts similars:
    for _, row in data.iterrows():
        id_d = row['id']
        label = row['label']
        similars = models.get_similar_from_text(id_d)
        print_info(data, id_d,label, similars[0])
        # MODEL 1:
        model_1_sim = similars[0]
        res = check_results(data, model_1_sim, id_d, label)
        result_m1_max.append(res[0])
        result_m1_mean.append(res[1])

        # MODEL 2:
        model_1_sim = similars[0]
        res = check_results(data, model_1_sim, id_d, label)
        result_m2_max.append(res[0])
        result_m2_mean.append(res[1])
    # INSERT IN DF
    data['result_m1_max'] = result_m1_max
    data['result_m1_mean'] = result_m1_mean
    data['result_m2_max'] = result_m2_max
    data['result_m2_mean'] = result_m2_mean


def check_results(data, similarity, id_d, label):
    similars = similarity[0:3]
    labels = get_sim_labels(data, similars)
    # Top 3: MAX
    max_value = get_max(labels)
    res_max = 'correct'
    if label != max_value:
        res_max = 'incorrect'
    # Top 3: MEAN
    mean_value = get_mean(labels)
    res_mean = 'correct' 
    if label != mean_value:
        res_mean = 'incorrect'
    return res_max, res_mean

"""
Get mean of three values and returns 0 or 1 depending of its proximity
"""
def get_mean(vals):
    sumt = 0
    for val in vals:
        sumt += val
    mean =  sumt  / 3
    if mean < 0.5:
        return 0
    else:
        return 1

"""
Returns the number with more repetitions
"""
def get_max(vals):
    ones = 0
    zeros = 0
    for val in vals:
        if val: # CHECK
            ones +=1
        else:
            zeros +=1
    # CHECK MAX
    if ones > zeros: 
        return 1
    else:
        return 0


def get_sim_labels(data, similars):
    list_labels = []
    for sim in similars:
        docid = sim[0]
        label = data[data['id'] == docid]
        list_labels.append(label['label'].values[0])
    return list_labels

def print_info(data, id_d,label, similars):
    fake = 'fake'
    if label == 0:
        fake = 'real'
    sim =  get_sim_labels(data, similars[0:3])
    print('Checking ID:', id_d, 'that is ', fake, 'and similars are: ', sim[0], sim[1], sim[2])

#https://github.com/RaRe-Technologies/movie-plots-by-genre/blob/master/ipynb_with_output/Document%20classification%20with%20word%20embeddings%20tutorial%20-%20with%20output.ipynb
#https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/