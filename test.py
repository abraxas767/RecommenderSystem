import unittest
from collections import Counter
import numpy as np
from tfidf import to_lower
from tfidf import remove_punctuation
from tfidf import tf
from tfidf import idf


class TestToLowerFunction(unittest.TestCase):
    def test_to_lower(self):
        self.assertEqual(to_lower("HELLO"), "hello")

class TestRemovePunctuation(unittest.TestCase):
    def test_remove_punctiuation(self):
        self.assertEqual(remove_punctuation("Hello!"), "Hello ")
        self.assertEqual(remove_punctuation("Hello!=*\""), "Hello    ")
        self.assertEqual(remove_punctuation("?.,/|\\+*#':;!$%&§(){}¢[]'\"@=<-_>\n"), "                                 ")

class TestTF(unittest.TestCase):
    def test_basic_case(self):
        input_array = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_array_almost_equal(tf(input_array), expected_output)

    def test_zero_case(self):
        input_array = np.array([0, 1, 2, 3, 4])
        expected_output = np.array([0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(tf(input_array), expected_output)

    def test_large_values(self):
        input_array = np.array([1000, 2000, 3000, 4000, 5000])
        expected_output = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_array_almost_equal(tf(input_array), expected_output)


class TestIDF(unittest.TestCase):
    def test_basic_case(self):
        d1 = Counter({'hello': 3, 'world': 2, 'test': 1}) 
        d2 = Counter({'hello': 3, 'mars': 2, 'body': 1}) 
        d3 = Counter({'sofa': 3, 'will': 2, 'be': 1}) 
        doc = Counter({'hello': 3, 'my': 2, 'test': 1}) 
        basic_vector = {'hello' : 0, 'world': 0, 'test': 0, 'mars': 0, 'body': 0, 'sofa': 0, 'will' : 0, 'be': 0, 'my': 0}
        corpus = [d1, d2, d3, doc]
        expected = np.array([0.287682, 0, 0.693147,0, 0, 0, 0, 0, 1.386294])
        np.testing.assert_array_almost_equal(idf(doc, corpus, basic_vector), expected)


class TestTFIDF(unittest.TestCase):
    def test_basic_case(self):
        pass

if __name__ == "__main__":
    unittest.main()




