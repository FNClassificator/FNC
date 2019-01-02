import pandas as pd
import json
from src.fake_news_detector.core.nlp import clean_text
from src.utils import io
from src.utils import log

# File that include function to manage pandas dataframe

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
	return get_dataframe_from_json(articles)



# Pre:  JSON content
# Post: Pandas dataframe with content input 
def get_dataframe_from_json(content):
    try:
        log.info('Generating DataFrame...')
        df = pd.DataFrame(data=content['articles'])
        log.info('Done')
        print(get_columns_name(df))
        df.info(memory_usage='deep')
        return df
    except ValueError as err:
        log.error('|DATAFRAME.py| - Can\'t create pandas dataframe')
        log.error(err)


# UTILS PANDAS FUNCTIONS
def get_columns_name(df):
    return df.dtypes

# rows x cols
def get_matrix_size(df):
    return df.shape