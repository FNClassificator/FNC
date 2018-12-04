import numpy as np  
import pandas as pd

from src.fake_news_detector.helpers.process_data import dataframe
from src.helpers.read_data import io

def modelate_dataset():
    # 1. Read data
    path = ''
    content = io.read_json_file(path)
    # 2. Create dataframe
    return dataframe.get_dataframe_from_json(content)


