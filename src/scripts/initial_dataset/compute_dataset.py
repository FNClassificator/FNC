from src.fake_news_detector.core.pre_process import extract_datasets as ef
from src.fake_news_detector.core.pre_process import raw_dataset as rd
from src.utils import io
# OBJECTIVE: Classificate by title
 

def compute_all():
    # 1. Get dataset
    dataset = rd.modelate_dataset()
    # 2. Clean title and extract features

    content_dataset = ef.get_content_dataset(dataset)
    io.write_json_file('src/data/dataset_content.json', content_dataset)
    #style_dataset = ef.get_style_dataset(dataset)
    #io.write_json_file('src/data/dataset_style.json', style_dataset)
    #similarity_dataset = ef.get_similarity_dataset(dataset)
    #io.write_json_file('src/data/dataset_similarity.json', similarity_dataset)

if __name__ == '__main__':
    compute_all()