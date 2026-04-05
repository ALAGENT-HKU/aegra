<<<<<<< HEAD:src/agent_server/middleware/__init__.py
from .double_encoded_json import DoubleEncodedJSONMiddleware
from .logger_middleware import StructLogMiddleware

__all__ = ["DoubleEncodedJSONMiddleware", "StructLogMiddleware"]
=======
from aegra_api.middleware.content_type_fix import ContentTypeFixMiddleware
from aegra_api.middleware.logger_middleware import StructLogMiddleware

__all__ = ["ContentTypeFixMiddleware", "StructLogMiddleware"]
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/middleware/__init__.py
