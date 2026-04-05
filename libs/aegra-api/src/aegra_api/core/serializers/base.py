"""Base serialization interface"""

from abc import ABC, abstractmethod
from typing import Any


class Serializer(ABC):
    """Abstract base class for object serialization"""

    @abstractmethod
    def serialize(self, obj: Any) -> Any:
        """Serialize an object to a JSON-compatible format"""
        pass


class SerializationError(Exception):
    """Raised when serialization fails"""

<<<<<<< HEAD:src/agent_server/core/serializers/base.py
    def __init__(
        self, message: str, obj_type: str, original_error: Exception | None = None
    ):
=======
    def __init__(self, message: str, obj_type: str, original_error: Exception | None = None):
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/core/serializers/base.py
        super().__init__(message)
        self.obj_type = obj_type
        self.original_error = original_error
