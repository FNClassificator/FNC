from src.fake_news_detector.helpers.read_data import dataframe
from src.utils import io
from src.fake_news_detector.helpers.nlp import clean_text


def modelate_dataset():
	articles = {
		'articles' : []
	}
	n_ini = 1
	n_fi = 116
	for x in range(n_ini,n_fi):
		path =  'src/data/articles_en/Article_' + str(x) + '.json'
		# Read file
		content = io.read_json_file(path)
		if content != None:
			articles['articles'].append(content)
	# Create dataframe
	return dataframe.get_dataframe_from_json(articles)


# CLEANING FUNCTIONS

# name: original column
def tokenize_by_word_and_clean(dataset, name):
	new_name = name + '_token_clean'
	dataset[new_name] = 'none' # Copy of column
	for index, row in dataset.iterrows():
		 text = clean_text.clean_text_words(row[name])
		 dataset[index][new_name] = text
