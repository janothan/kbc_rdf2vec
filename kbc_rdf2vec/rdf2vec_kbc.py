import os

import gensim
import logging
from gensim.models import KeyedVectors
from typing import List

from kbc_rdf2vec.dataset import DataSet

# noinspection PyArgumentList
logging.basicConfig(handlers=[logging.FileHandler(__file__ + '.log', 'w', 'utf-8'), logging.StreamHandler()],
                    format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG)


class Rdf2vecKbc:

    def __init__(self, model_path: str, data_set: DataSet, n: int = 10, file_for_predicate_exclusion: str = None):
        """Constructor

        Parameters
        ----------
        model_path : str
            A path to the gensim model file. The file can also be a keyed vector file with ending ".kv".
        data_set : DataSet
            The dataset for which the prediction shall be performed.
        n : int
            The number of predictions to make for each triple.
        file_for_predicate_exclusion : str
            The RDF2Vec model learns embeddings for h,l,t but cannot differentiate between them afterwards. Hence,
            when doing predictions for h and t, it may also predict l. If the file used to train the embedding is given
            here, such relations will be removed from the proposal set.
        """
        if not os.path.isfile(model_path):
            logging.error(f"Cannot find file: {model_path}\nCurrent working directory: {os.getcwd()}")

        if model_path.endswith(".kv"):
            print("Gensim vector file detected.")
            self._vectors = KeyedVectors.load(model_path, mmap='r')
        else:
            self._vectors = gensim.models.Word2Vec.load(model_path).wv

        self.n = n
        self.data_set = data_set
        self.test_set = self.data_set.test_set()

        self._predicates = set()
        if file_for_predicate_exclusion is not None and os.path.isfile(file_for_predicate_exclusion):
            self._predicates = self._read_predicates(file_for_predicate_exclusion)

    def _read_predicates(self, file_for_predicate_exclusion) -> set:
        """Obtain predicates from the given nt file.

        Parameters
        ----------
        file_for_predicate_exclusion : str
            The NT file which shall be checked for predicates.

        Returns
        -------
        set
            A set of predicates (str).
        """
        with open(file_for_predicate_exclusion, "r", encoding="utf8") as f:
            result_set = set()
            for line in f:
                tokens = line.split(sep=" ")
                result_set.add(self.remove_tags(tokens[1]))
            return result_set

    @staticmethod
    def remove_tags(string_to_process : str) -> str:
        """Removes tags around a string. Space-trimming is also applied.

        Parameters
        ----------
        string_to_process : str
            The string for which tags shall be removed.

        Returns
        -------
        str
            Given string without tags.

        """
        string_to_process = string_to_process.strip(" ")
        if string_to_process.startswith("<"):
            string_to_process = string_to_process[1:]
        if string_to_process.endswith(">"):
            string_to_process = string_to_process[:len(string_to_process)-1]
        return string_to_process

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
        result_with_confidence = self._remove_predicates(result_with_confidence)
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
        result_with_confidence = self._remove_predicates(result_with_confidence)
        result = [i[0] for i in result_with_confidence]
        return result

    def _remove_predicates(self, list_to_process: List) -> List:
        result = []
        for entry in list_to_process:
            if not entry[0] in self._predicates:
                result.append(entry)
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
    kbc = Rdf2vecKbc(model_path="./wn_vectors/model.kv", n=5000, data_set=DataSet.WN18,
                     file_for_predicate_exclusion="./wordnet_kbc.nt")
    kbc.predict("./wn_evaluation_file_5000.txt")
