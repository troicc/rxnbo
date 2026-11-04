import numpy as np

from rxnbo import bo, experiments


def test_tables_are_saved_in_a_fresh_clone(tmp_path, monkeypatch):
    # results/ does not exist yet: whichever figure runs first must make it
    monkeypatch.setattr(experiments, "R", tmp_path / "results")
    y = np.arange(100.0)
    experiments.table1([bo.Run("random", 0, [5, 50, 99, 98])], y)
    assert (tmp_path / "results" / "table1_bo.csv").exists()
