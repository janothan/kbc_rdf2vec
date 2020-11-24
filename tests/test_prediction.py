import numpy as np
from gensim.models import KeyedVectors

from kbc_rdf2vec.dataset import DataSet
from kbc_rdf2vec.prediction import (
    PredictionFunctionEnum,
    RandomPredictionFunction,
    AveragePredicateAdditionPredictionFunction,
)


class TestPredictionFunction:
    def test_get_instance(self) -> None:
        kv = KeyedVectors.load("./tests/test_resources/mini_3d_wn_model.kv", mmap="r")
        for function in PredictionFunctionEnum:
            assert (
                function.get_instance(keyed_vectors=kv, data_set=DataSet.WN18)
                is not None
            )

    def test_implementation_of_all_prediction_functions(self) -> None:
        """Test whether implementations exist and whether reflexivity is implemented."""
        kv = KeyedVectors.load("./tests/test_resources/mini_3d_wn_model.kv", mmap="r")
        for function in PredictionFunctionEnum:
            # forbid reflexive
            function_instance = function.get_instance(
                keyed_vectors=kv,
                data_set=DataSet.WN18,
                is_reflexive_match_allowed=False,
            )
            assert function_instance is not None
            result = function_instance.predict_heads(
                ["09590495", "_synset_domain_topic_of", "09689152"], n=None
            )
            assert "09689152" not in result
            result = function_instance.predict_tails(
                ["09590495", "_synset_domain_topic_of", "09689152"], n=None
            )
            assert "09590495" not in result

            # allow reflexive (but exclude most similar due to implementation of gensim excluding those always)
            if (
                function == PredictionFunctionEnum.MOST_SIMILAR
                or function == PredictionFunctionEnum.PREDICATE_AVERAGING_MOST_SIMILAR
            ):
                continue

            function_instance = function.get_instance(
                keyed_vectors=kv, data_set=DataSet.WN18, is_reflexive_match_allowed=True
            )
            assert function_instance is not None
            result = function_instance.predict_heads(
                ["09590495", "_synset_domain_topic_of", "09689152"], n=None
            )
            assert "09689152" in (item[0] for item in result)
            result = function_instance.predict_tails(
                ["09590495", "_synset_domain_topic_of", "09689152"], n=None
            )
            assert "09590495" in (item[0] for item in result)


class TestRandomPredictionFunction:
    def test_predict_heads(self):
        kv = KeyedVectors.load("./tests/test_resources/wn_test_model.kv", mmap="r")
        rpf = RandomPredictionFunction(
            keyed_vectors=kv, data_set=DataSet.WN18, is_reflexive_match_allowed=False
        )
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
        assert "09590495" not in result

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
        appf = AveragePredicateAdditionPredictionFunction(
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
