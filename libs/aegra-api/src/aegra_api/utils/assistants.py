from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid5

<<<<<<< HEAD:src/agent_server/utils/assistants.py
from ..constants import ASSISTANT_NAMESPACE_UUID


def resolve_assistant_id(
    requested_id: str, available_graphs: Mapping[str, object]
) -> str:
=======
from aegra_api.constants import ASSISTANT_NAMESPACE_UUID


def resolve_assistant_id(requested_id: str, available_graphs: Mapping[str, object]) -> str:
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/utils/assistants.py
    """Resolve an assistant identifier.

    If the provided identifier matches a known graph id, derive a
    deterministic assistant UUID using the project namespace. Otherwise,
    return the identifier as-is.

    Args:
        requested_id: The value provided by the client (assistant UUID or graph id).
        available_graphs: Graph registry mapping; only keys are used for membership.

    Returns:
        A string assistant_id suitable for DB lookups and FK references.
    """
<<<<<<< HEAD:src/agent_server/utils/assistants.py
    return (
        str(uuid5(ASSISTANT_NAMESPACE_UUID, requested_id))
        if requested_id in available_graphs
        else requested_id
    )
=======
    return str(uuid5(ASSISTANT_NAMESPACE_UUID, requested_id)) if requested_id in available_graphs else requested_id
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/utils/assistants.py
