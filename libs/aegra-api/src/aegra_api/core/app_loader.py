"""Custom application loader for dynamic FastAPI/Starlette app imports"""

import importlib
import importlib.util
from pathlib import Path

import structlog
<<<<<<< HEAD:src/agent_server/core/app_loader.py
from starlette.applications import Starlette
=======
from fastapi import FastAPI
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/core/app_loader.py

logger = structlog.get_logger(__name__)


<<<<<<< HEAD:src/agent_server/core/app_loader.py
def load_custom_app(app_import: str) -> Starlette | None:
    """Load custom Starlette/FastAPI app from import path.
=======
def load_custom_app(app_import: str, base_dir: Path | None = None) -> FastAPI | None:
    """Load custom FastAPI app from import path.
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/core/app_loader.py

    Supports both file-based and module-based imports:
    - File path: "./custom_routes.py:app" or "/path/to/file.py:app"
    - Module path: "my_package.custom:app"

    Args:
        app_import: Import path in format "path/to/file.py:variable" or "module.path:variable"
<<<<<<< HEAD:src/agent_server/core/app_loader.py

    Returns:
        Loaded Starlette/FastAPI app instance or None if path is invalid
=======
        base_dir: Base directory for resolving relative file paths (e.g., config file directory)

    Returns:
        Loaded FastAPI app instance or None if path is invalid
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/core/app_loader.py

    Raises:
        ImportError: If the module or file cannot be imported
        AttributeError: If the specified variable is not found in the module
<<<<<<< HEAD:src/agent_server/core/app_loader.py
        TypeError: If the loaded object is not a Starlette/FastAPI application
=======
        TypeError: If the loaded object is not a FastAPI application
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/core/app_loader.py
    """
    logger.info(f"Loading custom app from {app_import}")

    if ":" not in app_import:
        raise ValueError(
            f"Invalid app import path format: {app_import}. "
            "Expected format: 'path/to/file.py:variable' or 'module.path:variable'"
        )

    path, name = app_import.rsplit(":", 1)

    try:
        # Determine if it's a file path or module path
        path_obj = Path(path)
<<<<<<< HEAD:src/agent_server/core/app_loader.py
        is_file_path = path_obj.is_file() or path.endswith(".py")

        if is_file_path:
            # Import from file path
            if not path_obj.exists():
                raise FileNotFoundError(f"Custom app file not found: {path}")

            spec = importlib.util.spec_from_file_location(
                "custom_app_module", str(path)
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec from {path}")
=======
        is_file_path = path_obj.suffix == ".py" or path.startswith("./") or path.startswith("../")

        if is_file_path:
            # Resolve relative paths from base_dir if provided
            if not path_obj.is_absolute() and base_dir is not None:
                path_obj = (base_dir / path_obj).resolve()

            # Import from file path
            if not path_obj.exists():
                raise FileNotFoundError(f"Custom app file not found: {path_obj}")

            spec = importlib.util.spec_from_file_location("custom_app_module", str(path_obj))
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec from {path_obj}")
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/core/app_loader.py

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # Import as a normal module
            module = importlib.import_module(path)

        # Get the app instance from the module
        if not hasattr(module, name):
            raise AttributeError(
                f"App '{name}' not found in module '{path}'. "
                f"Available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}"
            )

        user_app = getattr(module, name)

<<<<<<< HEAD:src/agent_server/core/app_loader.py
        # Validate it's a Starlette/FastAPI application
        if not isinstance(user_app, Starlette):
            raise TypeError(
                f"Object '{name}' in module '{path}' is not a Starlette or FastAPI application. "
                "Please initialize your app by importing and using the appropriate class:\n"
                "from starlette.applications import Starlette\n\n"
                "app = Starlette(...)\n\n"
                "or\n\n"
                "from fastapi import FastAPI\n\n"
                "app = FastAPI(...)\n\n"
=======
        # Validate it's a FastAPI application
        if not isinstance(user_app, FastAPI):
            raise TypeError(
                f"Object '{name}' in module '{path}' is not a FastAPI application. "
                "Custom apps must be FastAPI instances for proper OpenAPI support.\n"
                "Please initialize your app using:\n\n"
                "from fastapi import FastAPI\n\n"
                "app = FastAPI()\n\n"
>>>>>>> origin/dev_ALAGENT-HKU-merged:libs/aegra-api/src/aegra_api/core/app_loader.py
            )

        logger.info(f"Successfully loaded custom app '{name}' from {path}")
        return user_app

    except ImportError as e:
        raise ImportError(f"Failed to import app module '{path}': {e}") from e
    except AttributeError as e:
        raise AttributeError(f"App '{name}' not found in module '{path}'") from e
