import pytest

from geometric_signatures.tasks import REQUIRED_TASKS, validate_task_battery


def test_validate_task_battery_accepts_required_tasks() -> None:
    received = validate_task_battery(REQUIRED_TASKS)
    assert received == REQUIRED_TASKS


def test_validate_task_battery_rejects_missing_task() -> None:
    with pytest.raises(ValueError, match="missing required tasks"):
        validate_task_battery((
            "context_dependent_integration",
            "evidence_accumulation",
            "working_memory",
        ))


def test_validate_task_battery_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="unknown tasks"):
        validate_task_battery((
            "context_dependent_integration",
            "evidence_accumulation",
            "working_memory",
            "perceptual_discrimination",
            "nonexistent_task",
        ))
