import unittest

from src.fake_news_detector.nlp.word2vec import Word2Vec
from src.fake_news_detector.nlp.features import quantity as q

class TestWord2ver(unittest.TestCase):

    def test_get_results_word2vec(self):
        documents = #TODO
        w2v = Word2Vec(documetns)
        w2v.train()
        word = 'many'
        result = w2v.similar(word)
        return

if __name__ == '__main__':
    unittest.main()