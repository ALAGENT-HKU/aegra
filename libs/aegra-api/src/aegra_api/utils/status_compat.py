"""Status validation utilities.

After migration, all database records have standard status values.
This module validates that statuses conform to the API specification.
"""

<<<<<<< HEAD:src/agent_server/utils/status_compat.py
from ..models.enums import RunStatus, ThreadStatus
=======
from aegra_api.models.enums import RunStatus, ThreadStatus
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/utils/status_compat.py


def validate_run_status(status: str) -> RunStatus:
    """Validate that run status conforms to API specification.

    After migration, all statuses should be standard values.
    This function validates and rejects any invalid statuses.

    Args:
        status: Status string to validate

    Returns:
        Validated RunStatus value

    Raises:
        ValueError: If status is not a valid RunStatus
    """
    valid_statuses: list[RunStatus] = [
        "pending",
        "running",
        "error",
        "success",
        "timeout",
        "interrupted",
    ]

    if status not in valid_statuses:
<<<<<<< HEAD:src/agent_server/utils/status_compat.py
        raise ValueError(
            f"Invalid run status: {status}. Must be one of: {valid_statuses}"
        )
=======
        raise ValueError(f"Invalid run status: {status}. Must be one of: {valid_statuses}")
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/utils/status_compat.py

    return status  # type: ignore


def validate_thread_status(status: str) -> ThreadStatus:
    """Validate that thread status conforms to API specification.

    Args:
        status: Thread status string to validate

    Returns:
        Validated ThreadStatus value

    Raises:
        ValueError: If status is not a valid ThreadStatus
    """
    valid_statuses: list[ThreadStatus] = ["idle", "busy", "interrupted", "error"]

    if status not in valid_statuses:
<<<<<<< HEAD:src/agent_server/utils/status_compat.py
        raise ValueError(
            f"Invalid thread status: {status}. Must be one of: {valid_statuses}"
        )
=======
        raise ValueError(f"Invalid thread status: {status}. Must be one of: {valid_statuses}")
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/utils/status_compat.py

    return status  # type: ignore
