"""Tests for tasks_data base classes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from geometric_signatures.tasks_data.base import TaskBatch, TaskDataset


class TestTaskBatch:
    """Tests for TaskBatch frozen dataclass."""

    def test_construction(self) -> None:
        batch = TaskBatch(
            inputs=torch.zeros(4, 10, 2),
            targets=torch.zeros(4, 10, 1),
            mask=torch.ones(4, 10),
            task_name="test_task",
            conditions={"coh": torch.zeros(4)},
            epoch_boundaries={"response": torch.full((4,), 8, dtype=torch.long)},
        )
        assert batch.task_name == "test_task"
        assert batch.inputs.shape == (4, 10, 2)
        assert batch.targets.shape == (4, 10, 1)
        assert batch.mask.shape == (4, 10)

    def test_frozen(self) -> None:
        batch = TaskBatch(
            inputs=torch.zeros(2, 5, 1),
            targets=torch.zeros(2, 5, 1),
            mask=torch.ones(2, 5),
            task_name="test_task",
            conditions={},
            epoch_boundaries={},
        )
        with pytest.raises(AttributeError):
            batch.task_name = "other"  # type: ignore[misc]

    def test_conditions_accessible(self) -> None:
        coh = torch.tensor([0.5, -0.3])
        batch = TaskBatch(
            inputs=torch.zeros(2, 5, 1),
            targets=torch.zeros(2, 5, 1),
            mask=torch.ones(2, 5),
            task_name="test",
            conditions={"coherence": coh},
            epoch_boundaries={},
        )
        assert torch.equal(batch.conditions["coherence"], coh)

    def test_epoch_boundaries_accessible(self) -> None:
        resp = torch.tensor([8, 8], dtype=torch.long)
        batch = TaskBatch(
            inputs=torch.zeros(2, 10, 1),
            targets=torch.zeros(2, 10, 1),
            mask=torch.ones(2, 10),
            task_name="test",
            conditions={},
            epoch_boundaries={"response": resp},
        )
        assert torch.equal(batch.epoch_boundaries["response"], resp)


class TestTaskDataset:
    """Tests for TaskDataset ABC."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            TaskDataset()  # type: ignore[abstract]

    def test_subclass_must_implement_generate_batch(self) -> None:
        class IncompleteTask(TaskDataset):
            task_name = "incomplete"
            input_dim = 1
            output_dim = 1

        with pytest.raises(TypeError):
            IncompleteTask()  # type: ignore[abstract]

    def test_valid_subclass(self) -> None:
        class DummyTask(TaskDataset):
            task_name = "dummy"
            input_dim = 1
            output_dim = 1

            @property
            def n_timepoints(self) -> int:
                return 10

            def generate_batch(
                self, batch_size: int, rng: torch.Generator
            ) -> TaskBatch:
                return TaskBatch(
                    inputs=torch.zeros(batch_size, 10, 1),
                    targets=torch.zeros(batch_size, 10, 1),
                    mask=torch.ones(batch_size, 10),
                    task_name=self.task_name,
                    conditions={},
                    epoch_boundaries={},
                )

        task = DummyTask()
        assert task.task_name == "dummy"
        rng = torch.Generator().manual_seed(0)
        batch = task.generate_batch(4, rng)
        assert batch.inputs.shape == (4, 10, 1)
