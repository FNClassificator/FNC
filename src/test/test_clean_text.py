import unittest

from src.fake_news_detector.nlp import clean_text as ct

class TestCleanText(unittest.TestCase):

    def test_clean_text_by_sentence(self):
        text = 'Here there is one sentence. Here the other'
        result = ct.clean_text_by_sentence(text)
        expected = ['here there be one sentence', 'here the other']
        self.assertEqual(result,expected)
        return
    
    def test_clean_text_by_word(self):
        text = 'Here there is one sentence. Here the other'
        result = ct.clean_text_by_word(text)
        expected = ['here', 'there', 'be', 'one', 'sentence', 'here', 'the', 'other']
        self.assertEqual(result,expected)
        return

if __name__ == '__main__':
    unittest.main()