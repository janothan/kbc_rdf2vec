import gensim
import logging
from gensim.models import KeyedVectors
from typing import List

from kbc_rdf2vec.dataset import DataSet

# noinspection PyArgumentList
logging.basicConfig(handlers=[logging.FileHandler(__file__ + '.log', 'w', 'utf-8'), logging.StreamHandler()],
                    format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)


class Rdf2vecKbc:

    def __init__(self, model_path: str, data_set: DataSet, n: int = 10):
        """Constructor

        Parameters
        ----------
        model_path : str
            A path to the gensim model file. The file can also be a keyed vector file with ending ".kv".
        data_set : DataSet
            The dataset for which the prediction shall be performed.
        n : int
            The number of predictions to make for each triple.
        """

        if model_path.endswith(".kv"):
            print("Gensim vector file detected.")
            self._vectors = KeyedVectors.load(model_path, mmap='r')
        else:
            self._vectors = gensim.models.Word2Vec.load(model_path).wv

        self.n = n
        self.data_set = data_set
        self.test_set = self.data_set.test_set()

    def predict(self, file_to_write: str):
        """Performs the actual predictions. A file will be generated.

        Parameters
        ----------
        file_to_write : str
            File that shall be written for further evaluation.
        """

        with open(file_to_write, "w+", encoding="utf8") as f:
            erroneous_triples = 0
            for triple in self.test_set:
                logging.info(f"Processing triple: {triple}")
                if self._check_triple(triple):
                    f.write(f"{triple[0]} {triple[1]} {triple[2]}\n")
                    heads = self._predict_heads(triple)
                    tails = self._predict_tails(triple)
                    f.write(f"\tHeads: {' '.join(heads)}\n")
                    f.write(f"\tTails: {' '.join(tails)}\n")
                else:
                    logging.error(f"Could not process the triple: {triple}")
                    erroneous_triples += 1

            # logging output for the user
            if erroneous_triples == 0:
                logging.info("Erroneous Triples: " + str(erroneous_triples))
            else:
                logging.error("Erroneous Triples: " + str(erroneous_triples))

    def _predict_heads(self, triple: List[str]) -> List[str]:
        """Predicts n heads given a triple.

        Parameters
        ----------
        triple : List[str]
            The triple for which n heads shall be predicted.

        Returns
        -------
        List[str]
            A list of predicted concepts.
        """
        result_with_confidence = self._vectors.most_similar(positive=list(triple[1:]), topn=self.n)
        result = [i[0] for i in result_with_confidence]
        return result

    def _predict_tails(self, triple: List[str]) -> List[str]:
        """Predicts n tails given a triple

        Parameters
        ----------
        triple: List[str]
            The triple for which n tails shall be predicted.

        Returns
        -------
        List[str]
            A list of predicted concepts.
        """
        result_with_confidence = self._vectors.most_similar(positive=list(triple[:2]), topn=self.n)
        result = [i[0] for i in result_with_confidence]
        return result

    def _check_triple(self, triple: List[str]) -> bool:
        """Triples can only be processed if all three elements are available in the vector space. This methods
        checks for exactly this.

        Parameters
        ----------
        triple : List[str]
            The triple that shall be checked.

        Returns
        -------
        bool
            True if all three elements of the triple exist in the given vector space, else False.
        """
        try:
            self._vectors.get_vector(triple[0])
            self._vectors.get_vector(triple[1])
            self._vectors.get_vector(triple[2])
            return True
        except KeyError:
            return False


if __name__ == "__main__":
    kbc = Rdf2vecKbc(model_path="../wn_vectors/model.kv", n=10, data_set=DataSet.WN18)
    kbc.predict("./wn_evaluation_file.txt")
