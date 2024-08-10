import unittest
from collections import Counter
import numpy as np
from tfidf import get_basic_vector, preprocess, preprocess_all, to_lower
from tfidf import remove_punctuation
from tfidf import tf
from tfidf import idf
from tfidf import tfidf

test_korpus = [
    "The CAT cat cat sat sat on the mat and the mat was blue @ 3 PM.",
    "The dog dog dog chased the CAT, but the cat climbed the tree!!!",
    "Birds flew over the TREE tree tree, and the DOG watched the bird...",
    "The dog chased the CAT, but the cat climbed the tree!!!",
    "The fish fish swam in the pond while the fish sat on the TREE at 5 PM."]

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

    def test_test_corpus(self):
        b = preprocess(test_korpus[0]) 
        expected_output = np.array([1.0, 0.6666666, 0.6666666, 0.3333333, 0.3333333])
        np.testing.assert_array_almost_equal(tf(np.array(list(b.values()))), expected_output)


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

    def test_test_corpus(self):
        c = preprocess_all(test_korpus)
        base = get_basic_vector(c)
        tfidf(c[0], c, base)


class TestTFIDF(unittest.TestCase):
    def test_basic_case(self):
        pass

if __name__ == "__main__":
    unittest.main()




