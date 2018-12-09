from src.fake_news_detector.pre_process import extact_features as ef
from src.fake_news_detector.read_data import dataframe

# OBJECTIVE: Classificate by title
 

def compute_all():
    # 1. Get dataset
    dataset = dataframe.modelate_dataset()
    # 2. Clean title and extract features
    #ef.get_title_info(dataset)
    #ef.get_similarity_info(dataset)
    ef.get_text_info(dataset)

if __name__ == '__main__':
    compute_all()