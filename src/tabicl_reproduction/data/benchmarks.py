"""OpenML ids of the benchmark datasets, kept in code for reproducibility.

To add a benchmark, define its {name: openml_id} dict here and register it in
BENCHMARKS. tests/data/test_benchmarks.py checks the ids against OpenML.
"""

# Classification development datasets from Supplementary Table 5 of
# Hollmann et al., "Accurate predictions on small data with a tabular
# foundation model", Nature 637 (2025) (TabPFNv2). These are the validation
# datasets used in the TabICLv2 ablations.
# Paper: https://doi.org/10.1038/s41586-024-08328-6
# Supplementary PDF with the table:
# https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-08328-6/MediaObjects/41586_2024_8328_MOESM2_ESM.pdf  # noqa: E501

TABPFN_VAL_DATASETS: dict[str, int] = {
    "adaprior": 1037,
    "allbp": 40707,
    "analcatdata_asbestos": 459,
    "analcatdata_cyyoung8092": 465,
    "ar3": 1060,
    "ar6": 1064,
    "baseball": 185,
    "birds": 40588,
    "blogger": 1463,
    "breast-tissue": 1465,
    "cjs": 473,
    "clean1": 40665,
    "cleveland-nominal": 40711,
    "compas-two-years": 42192,
    "confidence": 468,
    "corral": 40669,
    "delta_ailerons": 803,
    "dermatology": 35,
    "eye_movements": 44130,
    "GAMETES_Epistasis": 40646,
    "grub-damage": 338,
    "heart-c": 49,
    "heart-h": 51,
    "heart-statlog": 53,
    "heloc": 45026,
    "hill-valley": 1479,
    "ionosphere": 59,
    "JapaneseVowels": 375,
    "jungle_chess_2pcs_endgame_panther_lion": 40997,
    "led24": 40677,
    "mc1": 1056,
    "Midwest_Survey_nominal": 42532,
    "mw1": 1071,
    "national-longitudinal-survey-binary": 43892,
    "olindda_outliers": 42793,
    "page-blocks": 30,
    "penguins": 42585,
    "regime_alimentaire": 42172,
    "ringnorm": 1496,
    "rl": 44160,
    "rmftsa_sleepdata": 679,
    "scene": 312,
    "Smartphone-Based_Recognition": 4153,
    "squash-stored": 340,
    "synthetic_control": 377,
    "teachingAssistant": 1115,
    "thyroid-allbp": 40474,
    "thyroid-ann": 40497,
    "Titanic": 40704,
    "tokyo1": 40705,
    "Tour-and-Travels-Churn": 45545,
    "user-knowledge": 1508,
    "volcanoes-a1": 1527,
    "vote": 56,
    "waveform-5000": 60,
    "wholesale-customers": 1511,
    "xd6": 40693,
}


# Benchmark name (as used by load_data) -> {dataset name: OpenML id}.
BENCHMARKS: dict[str, dict[str, int]] = {
    "tabpfn_val": TABPFN_VAL_DATASETS,
}
