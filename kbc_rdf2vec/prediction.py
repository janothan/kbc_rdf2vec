import logging
from enum import Enum
from random import randint, random
from typing import List, Tuple

from gensim.models import KeyedVectors

from kbc_rdf2vec.dataset import DataSet

# noinspection PyArgumentList
logging.basicConfig(
    handlers=[logging.FileHandler(__file__ + ".log", "w", "utf-8")],
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.DEBUG,
)


class PredictionFunctionInterface:
    """Interface for prediction functions."""

    def __init__(self, keyed_vectors: KeyedVectors, data_set: DataSet):
        self._keyed_vectors = keyed_vectors
        self._data_set = data_set

    def predict_heads(self, triple: List[str], n: int) -> List[Tuple[str, float]]:
        pass

    def predict_tails(self, triple: List[str], n: int) -> List[Tuple[str, float]]:
        pass


class PredictionFunction(Enum):
    """An enumeration with the implemented similarity functions."""

    MOST_SIMILAR = "most_similar"
    RANDOM = "random"

    def get_instance(
        self, keyed_vectors, data_set: DataSet
    ) -> PredictionFunctionInterface:
        """Obtain the accompanying instance.

        Parameters
        ----------
        data_set: DataSet
            The dataset to be evaluated.
        keyed_vectors
            Keyed vectors instance for which the similarity shall be applied.

        Returns
        -------
        PredictionFunctionInterface
            An instance of the PredictionFunctionInterface.
        """
        if self.value == "most_similar":
            return MostSimilarPredictionFunction(
                keyed_vectors=keyed_vectors, data_set=data_set
            )
        if self.value == "random":
            return RandomPredictionFunction(
                keyed_vectors=keyed_vectors, data_set=data_set
            )


class RandomPredictionFunction(PredictionFunctionInterface):
    """This class randomly picks results for h and t."""

    def predict_heads(self, triple: List[str], n: int) -> List[Tuple[str, float]]:
        vocab_size = len(self._keyed_vectors.vocab)
        if n is None:
            n = vocab_size
        if n > vocab_size:
            logging.error(
                f"n ({n}) > vocab_size ({vocab_size})! Predicting only {vocab_size} concepts."
            )
            n = vocab_size
        result_indices = set()
        if n != vocab_size:
            # run a (rather slow) drawing algorithm
            while len(result_indices) < n:
                result_indices.add(randint(0, vocab_size - 1))
        else:
            # scramble the list
            range_list = range(0, vocab_size - 1)
            result_indices = sorted(range_list, key=lambda x: random())
        result = []
        for index in result_indices:
            result.append((self._keyed_vectors.index2word[index], random()))

        return result

    def predict_tails(self, triple: List[str], n: int) -> List[Tuple[str, float]]:
        """In the random case, there is no difference between predict_heads and predict_tails.

        Parameters
        ----------
        triple: List[str]
            Triple for which the tails shall be predicted.
        n: int
            Number of prediction to make.

        Returns
        -------
        List[Tuple[str, float]
            List of predictions with confidences. Note that in this case the confidences are random floats.
        """
        return self.predict_heads(triple=triple, n=n)


class MostSimilarPredictionFunction(PredictionFunctionInterface):
    """This class simply calls the gensim "most_similar" function with (h,l) to predict t and with (l,t) to predict
    h.
    """

    def predict_heads(self, triple: List[str], n: int) -> List[Tuple[str, float]]:
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=list(triple[1:]), topn=n
        )
        # important: if self.n is none, the result type of the most_similar action is a numpy array that needs to be
        # mapped manually.
        if n is None:
            new_result_with_confidence = []
            assert len(result_with_confidence) == len(self._keyed_vectors.vocab)
            for i, similarity in enumerate(result_with_confidence):
                word = self._keyed_vectors.index2word[i]
                if word != triple[1] and word != triple[2]:
                    # avoid predicting the inputs
                    new_result_with_confidence.append((word, similarity))
            result_with_confidence = sorted(
                new_result_with_confidence, key=lambda x: x[1], reverse=True
            )
        return result_with_confidence

    def predict_tails(self, triple: List[str], n: int) -> List[Tuple[str, float]]:
        result_with_confidence = self._keyed_vectors.most_similar(
            positive=list(triple[:2]), topn=n
        )
        # important: if self.n is none, the result type of the most_similar action is a numpy array that needs to be
        # mapped manually.
        if n is None:
            new_result_with_confidence = []
            assert len(result_with_confidence) == len(self._keyed_vectors.vocab)
            for i, similarity in enumerate(result_with_confidence):
                word = self._keyed_vectors.index2word[i]
                if word != triple[0] and word != triple[1]:
                    # avoid predicting the inputs
                    new_result_with_confidence.append((word, similarity))
            result_with_confidence = sorted(
                new_result_with_confidence, key=lambda x: x[1], reverse=True
            )
        return result_with_confidence


class AveragePredicatePredictionFunction(PredictionFunctionInterface):
    def __init__(self, keyed_vectors: KeyedVectors, data_set: DataSet):
        super().__init__(keyed_vectors=keyed_vectors, data_set=data_set)

        # now we build a dictionary from predicate to (subject, object)
        all_triples = []
        all_triples.extend(data_set.valid_set())
        all_triples.extend(data_set.train_set())

        p_to_so = {}
        for triple in all_triples:
            if p_to_so[triple[1]] is None:
                p_to_so[triple[1]] = {triple[0], triple[2]}
            else:
                p_to_so[triple[1]].add(triple[0])
                p_to_so[triple[1]].add(triple[1])
