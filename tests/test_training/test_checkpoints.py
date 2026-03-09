"""Tests for checkpoint save/load."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from geometric_signatures.training.checkpoints import load_checkpoint, save_checkpoint


class TestCheckpoints:
    """Tests for save_checkpoint and load_checkpoint."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / "ckpt.pt"
        save_checkpoint(model, optimizer, epoch=5, metrics={"loss": 0.1}, path=path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / "nested" / "dir" / "ckpt.pt"
        save_checkpoint(model, optimizer, epoch=0, metrics={}, path=path)
        assert path.exists()

    def test_round_trip(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())

        # Do a forward/backward to populate optimizer state
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Save
        path = tmp_path / "ckpt.pt"
        save_checkpoint(model, optimizer, epoch=3, metrics={"loss": 0.5}, path=path)

        # Load into fresh model
        model2 = torch.nn.Linear(4, 2)
        optimizer2 = torch.optim.Adam(model2.parameters())
        info = load_checkpoint(path, model2, optimizer2)

        assert info["epoch"] == 3
        assert info["metrics"]["loss"] == 0.5

        # Weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

    def test_load_without_optimizer(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / "ckpt.pt"
        save_checkpoint(model, optimizer, epoch=1, metrics={}, path=path)

        model2 = torch.nn.Linear(4, 2)
        info = load_checkpoint(path, model2)  # no optimizer
        assert info["epoch"] == 1

    def test_load_missing_file(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(4, 2)
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt", model)
