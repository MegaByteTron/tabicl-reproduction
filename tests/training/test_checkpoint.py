import os
import signal

import torch

from tabicl_reproduction.training import Checkpointer


def test_resume_continues_the_run(make_trainer, tables, tmp_path):
    trainer = make_trainer()
    checkpointer = Checkpointer(tmp_path)
    trainer.fit(tables(), n_epochs=1, steps_per_epoch=2, checkpointer=checkpointer)

    resumed = make_trainer()
    Checkpointer(tmp_path / "next").load(resumed, tmp_path)  # a run dir works
    assert (resumed.step, resumed.epoch) == (2, 1)

    resumed.fit(tables(), n_epochs=2, steps_per_epoch=2)
    assert (resumed.step, resumed.epoch) == (4, 2)


def test_fit_writes_last_and_model(make_trainer, tables, tmp_path):
    make_trainer().fit(
        tables(), n_epochs=1, steps_per_epoch=1, checkpointer=Checkpointer(tmp_path)
    )
    assert (tmp_path / "last.pt").exists()
    assert set(torch.load(tmp_path / "model.pt", weights_only=True)) == {
        "model",
        "step",
    }


def test_best_follows_the_monitored_metric(make_trainer, tmp_path):
    trainer = make_trainer()
    checkpointer = Checkpointer(tmp_path, monitor="eval/log_loss", mode="min")

    assert checkpointer.update_best(trainer, {"eval/log_loss": 0.5})
    assert not checkpointer.update_best(trainer, {"eval/log_loss": 0.7})
    assert not checkpointer.update_best(trainer, {"other": 0.1})
    assert checkpointer.update_best(trainer, {"eval/log_loss": 0.4})

    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert best["eval/log_loss"] == 0.4


def test_maybe_save_waits_for_the_interval(make_trainer, tmp_path):
    trainer = make_trainer()
    assert not Checkpointer(tmp_path, every_min=60).maybe_save(trainer)
    assert Checkpointer(tmp_path, every_min=0).maybe_save(trainer)
    assert (tmp_path / "last.pt").exists()


def test_slurm_signal_stops_and_saves(make_trainer, tables, tmp_path):
    trainer = make_trainer()
    checkpointer = Checkpointer(tmp_path)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR2, signal.SIGINT)}
    try:
        checkpointer.install_signal_handlers(trainer)
        os.kill(os.getpid(), signal.SIGUSR2)  # what SLURM sends before the limit
        status = trainer.fit(
            tables(), n_epochs=1, steps_per_epoch=3, checkpointer=checkpointer
        )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

    assert status == "stopped"
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "model.pt").exists()  # the weights so far, for evaluation
