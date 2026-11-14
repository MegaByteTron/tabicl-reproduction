import sys

import pandas as pd
import torch
from hydra_zen import save_as_yaml

from tabicl_reproduction import score
from tabicl_reproduction.data import PriorDataset, TableSample
from tabicl_reproduction.model.model import NanoTabICLv2
from tabicl_reproduction.score import predict_proba


def small_table():
    return PriorDataset(n_samples=60, max_features=6, max_classes=4).sample()


def test_one_order_is_a_plain_forward_pass(model):
    table = small_table()
    n_classes = int(table.y_train.max()) + 1
    x = torch.cat([table.X_train, table.X_test]).unsqueeze(0)
    with torch.no_grad():
        logits = model(x, table.y_train.unsqueeze(0))[0, :, :n_classes]
    expected = torch.softmax(logits, dim=-1)
    assert torch.allclose(predict_proba(model, table, n_orders=1), expected)


def test_orders_are_averaged(model):
    table = small_table()
    n_classes = int(table.y_train.max()) + 1
    proba = predict_proba(model, table)
    assert proba.shape == (len(table.X_test), n_classes)
    assert torch.allclose(proba.sum(dim=1), torch.ones(len(table.X_test)))
    assert torch.equal(proba, predict_proba(model, table))  # reproducible
    assert not torch.allclose(proba, predict_proba(model, table, n_orders=1))


def test_score_command(tmp_path, monkeypatch):
    # A fake training run: the resolved config and best.pt, as training writes them.
    run_dir = tmp_path / "run"
    (run_dir / ".hydra").mkdir(parents=True)
    sizes = {"embed_dim": 16, "col_num_blocks": 1, "row_num_blocks": 1}
    model = NanoTabICLv2(max_classes=10, out_dim=10, icl_num_blocks=1, **sizes)
    torch.save({"model": model.state_dict(), "step": 7}, run_dir / "best.pt")
    config = {
        "_target_": "tabicl_reproduction.model.model.NanoTabICLv2",
        "max_classes": 10,
        "out_dim": "${.max_classes}",  # as in a real run's config
        "icl_num_blocks": 1,
        **sizes,
    }
    save_as_yaml({"model": config}, run_dir / ".hydra" / "config.yaml")

    def fake_result_tables(limit):
        info = {"source": "prior_val", "split_seed": 0, "data_version": "v2"}
        tables = [TableSample(*small_table(), dataset_id=i) for i in range(2)]
        X_train, y_train, X_test, y_test = small_table()
        one_class = TableSample(X_train, 0 * y_train, X_test, 0 * y_test, 2)
        yield info, [*tables, one_class]

    monkeypatch.setattr(score, "result_tables", fake_result_tables)
    out = tmp_path / "results"
    argv = ["score", str(run_dir), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    score.main()

    rows = pd.read_csv(out / "tabicl_best.csv")
    assert rows["table_id"].tolist() == ["prior_0000", "prior_0001", "prior_0002"]
    assert rows["status"].tolist() == ["ok", "ok", "skipped"]  # one class
    assert (rows["model"] == "tabicl_best").all()
    assert (rows["checkpoint_step"] == 7).all()
    assert (out / "summary.txt").exists()
