import unittest

from src.fake_news_detector.nlp.features import words as w
from src.fake_news_detector.nlp.features import quantity as q

class TestWords(unittest.TestCase):

    def test_get_common_nouns(self):
        text = ['hello', 'my', 'name', 'is', 'Elena', 'bottle', 'bottle']
        tagged_text = q.get_tags(text)
        result = w.get_type_words(tagged_text,'N')
        expected = ['hello','name', 'Elena', 'bottle', 'bottle']
        self.assertEqual(result,expected)
        return

    def test_get_unique_common_nouns(self):
        text = ['hello', 'my', 'name', 'is', 'Elena', 'bottle', 'bottle']
        tagged_text = q.get_tags(text)
        result = w.get_unique_type_words(tagged_text,'N')
        expected = ['hello','name', 'Elena', 'bottle']
        self.assertEqual(result,expected)
        return
    
    def test_get_adj_words(self):
        text = ['my', 'bottle','and', 'is', 'pretty', 'cool', 'and', 'beautiful']
        tagged_text = q.get_tags(text)
        result = w.get_type_words(tagged_text,'J')
        expected = ['pretty','cool','beautiful']
        self.assertEqual(result,expected)
        return
    
    def test_get_unique_adj_words(self):
        text = ['my', 'bottle','and', 'is', 'pretty', 'cool', 'and', 'beautiful']
        tagged_text = q.get_tags(text)
        result = w.get_unique_type_words(tagged_text,'J')
        expected = ['pretty','cool','beautiful']
        self.assertEqual(result,expected)
        return

if __name__ == '__main__':
    unittest.main()