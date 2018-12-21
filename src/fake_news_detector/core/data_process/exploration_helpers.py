
""" 
Helpers functions to process the data 
on exploration and classification
"""

# Split data in three parts by value
def get_third_parts(min_i,max_i,data):
    third = (max_i - min_i) / 3
    th_one = min_i + third
    th_two = max_i - third
    first_f = 0
    second_f = 0
    third_f = 0
    for i in data:
        if i <= th_one:
            first_f += 1
        elif i >= th_two:
            third_f += 1
        else:
            second_f +=1
    return first_f, second_f, third_f

def most_influencial_words(model_log, vectorizer, genre_index=0, num_words=0):
    features = vectorizer.get_feature_names()
    max_coef = sorted(enumerate(model_log.coef_[genre_index]), key=lambda x:x[1], reverse=True)
    return [features[x[0]] for x in max_coef[:num_words]]