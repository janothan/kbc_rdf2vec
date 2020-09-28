from gensim.models import KeyedVectors

from kbc_rdf2vec.prediction import PredictionFunction, RandomPredictionFunction


class TestPredictionFunction:
    def test_get_instance(self):
        for function in PredictionFunction:
            assert function.get_instance(keyed_vectors=None) is not None


class TestRandomPredictionFunction:
    def test_predict_heads(self):
        kv = KeyedVectors.load("./tests/test_resources/wn_test_model.kv", mmap='r')
        rpf = RandomPredictionFunction(keyed_vectors=kv)
        result = rpf.predict_heads(['09590495', '_synset_domain_topic_of', '09689152'], n=10)
        assert len(result) == 10
        result = rpf.predict_heads(['09590495', '_synset_domain_topic_of', '09689152'], n=18)
        assert len(result) == 18

    def test_predict_tails(self):
        kv = KeyedVectors.load("./tests/test_resources/wn_test_model.kv", mmap='r')
        rpf = RandomPredictionFunction(keyed_vectors=kv)
        result = rpf.predict_tails(['09590495', '_synset_domain_topic_of', '09689152'], n=10)
        assert len(result) == 10
        result = rpf.predict_tails(['09590495', '_synset_domain_topic_of', '09689152'], n=23)
        assert len(result) == 23
