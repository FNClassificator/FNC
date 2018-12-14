import unittest

from src.fake_news_detector.nlp.features import subject as s
from src.fake_news_detector.nlp.features import quantity as q

class TestSubject(unittest.TestCase):

    def test_get_subject(self):
        text = 'My name is Elena'
        result = s.get_subject(text)
        expected = ['My', 'name']
        self.assertEqual(result,expected)
        return

if __name__ == '__main__':
    unittest.main()