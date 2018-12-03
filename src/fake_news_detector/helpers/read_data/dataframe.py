import pandas as pd
import json

from src.utils import log

# File that include function to manage pandas dataframe

# Pre:  JSON content
# Post: Pandas dataframe with content input 
def get_dataframe_from_json(content):
    try:
        log.info('Generating DataFrame...')
        df = pd.DataFrame(data=content['articles'])
        log.info('Done')
        print(get_columns_name(df))
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