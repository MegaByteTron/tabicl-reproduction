import pytest

from tabicl_reproduction.data import BENCHMARKS, TABPFN_VAL_DATASETS

# Names the paper shortens, mapped to the full name on OpenML.
OPENML_NAMES = {
    "adaprior": "ada_prior",
    "GAMETES_Epistasis": "GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1",
    "Smartphone-Based_Recognition": (
        "Smartphone-Based_Recognition_of_Human_Activities"
    ),
    "Tour-and-Travels-Churn": "Tour-and-Travels-Customer-Churn-Prediction",
}


def test_tabpfn_val_is_registered():
    assert BENCHMARKS["tabpfn_val"] is TABPFN_VAL_DATASETS


@pytest.mark.parametrize("name", BENCHMARKS)
def test_ids_are_unique(name):
    ids = list(BENCHMARKS[name].values())
    assert len(ids) == len(set(ids))


@pytest.mark.network
@pytest.mark.parametrize("name", BENCHMARKS)
def test_ids_match_openml(name):
    """Catch typos in the ids by comparing them with OpenML's metadata.

    Run with: uv run pytest -m network
    """
    import openml

    datasets = BENCHMARKS[name]
    info = openml.datasets.list_datasets(
        data_id=list(datasets.values()), output_format="dataframe"
    ).set_index("did")

    for our_name, dataset_id in datasets.items():
        assert dataset_id in info.index, f"{our_name}: id {dataset_id} not on OpenML"
        row = info.loc[dataset_id]
        expected = OPENML_NAMES.get(our_name, our_name)
        assert row["name"] == expected, (
            f"{our_name}: id {dataset_id} is {row['name']!r} on OpenML"
        )
        assert 2 <= row["NumberOfClasses"] <= 10, f"{our_name}: not 2-10 classes"
        assert row["NumberOfInstances"] <= 10_000, f"{our_name}: over 10k rows"
