import numpy as np
from gensim.models import KeyedVectors

from kbc_rdf2vec.dataset import DataSet
from kbc_rdf2vec.prediction import (
    PredictionFunction,
    RandomPredictionFunction,
    AveragePredicatePredictionFunction,
)


class TestPredictionFunction:
    def test_get_instance(self):
        for function in PredictionFunction:
            assert (
                function.get_instance(keyed_vectors=None, data_set=DataSet.WN18)
                is not None
            )


class TestRandomPredictionFunction:
    def test_predict_heads(self):
        kv = KeyedVectors.load("./tests/test_resources/wn_test_model.kv", mmap="r")
        rpf = RandomPredictionFunction(keyed_vectors=kv, data_set=DataSet.WN18)
        result = rpf.predict_heads(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=10
        )
        assert len(result) == 10
        result = rpf.predict_heads(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=18
        )
        assert len(result) == 18
        result = rpf.predict_heads(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=None
        )
        assert len(result) > 100

    def test_predict_tails(self):
        kv = KeyedVectors.load("./tests/test_resources/wn_test_model.kv", mmap="r")
        rpf = RandomPredictionFunction(keyed_vectors=kv, data_set=DataSet.WN18)
        result = rpf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=10
        )
        assert len(result) == 10
        result = rpf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=23
        )
        assert len(result) == 23
        result = rpf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=None
        )
        assert len(result) > 100


class TestAveragePredicatePredictionFunction:
    def test_constructor(self):
        kv = KeyedVectors.load("./tests/test_resources/mini_3d_wn_model.kv", mmap="r")
        appf = AveragePredicatePredictionFunction(
            keyed_vectors=kv, data_set=DataSet.WN18
        )
        # just checking whether the constructor works:
        assert appf is not None

        # test tail prediction n = 15
        result = appf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=15
        )
        assert len(result) == 15

        # test tail prediction n = all
        result = appf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=None
        )
        assert len(result) > 100

        result = appf.predict_heads(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=15
        )
        assert len(result) == 15

        # test head prediction n = all
        result = appf.predict_tails(
            ["09590495", "_synset_domain_topic_of", "09689152"], n=None
        )
        assert len(result) > 100

    def test_axioms_np(self):
        """Simple axioms to ensure correct usage."""
        my_list = [np.array([1, 2, 3]), np.array([3, 4, 5]), np.array([5, 6, 7])]
        mean_vector = np.mean(my_list, axis=0)
        assert mean_vector[0] == 3
        assert mean_vector[1] == 4
        assert mean_vector[2] == 5
