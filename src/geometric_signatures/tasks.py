from __future__ import annotations

REQUIRED_TASKS = (
    "context_dependent_integration",
    "evidence_accumulation",
    "working_memory",
    "perceptual_discrimination",
)


def validate_task_battery(tasks: tuple[str, ...]) -> tuple[str, ...]:
    received = tuple(tasks)
    missing = [task for task in REQUIRED_TASKS if task not in received]
    if missing:
        raise ValueError(f"Task battery missing required tasks: {', '.join(missing)}")
    unknown = [task for task in received if task not in REQUIRED_TASKS]
    if unknown:
        raise ValueError(f"Task battery contains unknown tasks: {', '.join(unknown)}")
    return received
