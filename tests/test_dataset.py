from kbc_rdf2vec.dataset import DataSet
import os.path


def test_files_exist():
    """The following is tested
    - Existence of test/valid/train files for every data set in enum DataSet
    """
    for data_set in DataSet:
        test_path = data_set.test_set_path()
        valid_path = data_set.valid_set_path()
        train_path = data_set.train_set_path()
        assert test_path is not None
        assert valid_path is not None
        assert train_path is not None
        assert os.path.isfile(test_path)
        assert os.path.isfile(valid_path)
        assert os.path.isfile(train_path)


def test_file_parser():
    """The following is tested
    - whether the given files can be parsed.
    """
    for data_set in DataSet:
        test_data = data_set.test_set()
        _assert_triples_not_none(test_data)
        train_data = data_set.train_set()
        _assert_triples_not_none(train_data)
        valid_data = data_set.valid_set()
        _assert_triples_not_none(valid_data)
        # making sure that different files were read:
        assert test_data[0][0] != train_data[0][0] or test_data[0][1] != train_data[0][1] \
               or test_data[0][2] != train_data[0][2]
        assert train_data[0][0] != valid_data[0][0] or train_data[0][1] != valid_data[0][1] \
               or train_data[0][2] != valid_data[0][2]


def _assert_triples_not_none(parsed_triples):
    """Simply runs a couple of assert statements for the given parsed triples.

    Parameters
    ----------
    parsed_triples
        The list of triples to be evaluated.

    """
    assert len(parsed_triples) > 10
    assert parsed_triples[0][0] is not None
    assert parsed_triples[0][1] is not None
    assert parsed_triples[0][2] is not None
