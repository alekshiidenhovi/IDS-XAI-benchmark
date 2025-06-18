import os
import neptune
import typing as T
from dotenv import load_dotenv


def parse_project_name() -> str:
    load_dotenv()
    workspace = os.getenv("NEPTUNE_WORKSPACE")
    if not workspace:
        raise ValueError("NEPTUNE_WORKSPACE is not set")
    project_name = os.getenv("NEPTUNE_PROJECT")
    if not project_name:
        raise ValueError("NEPTUNE_PROJECT is not set")
    project = f"{workspace}/{project_name}"
    return project


def get_neptune_api_token() -> str:
    load_dotenv()
    api_token = os.getenv("NEPTUNE_API_KEY")
    if not api_token:
        raise ValueError("NEPTUNE_API_KEY is not set")
    return api_token


def init_neptune_run(
    experiment_name: T.Optional[str] = None, run_id: T.Optional[str] = None
):
    """
    Initialize and configure a Neptune run.

    This function loads environment variables from a .env file, authenticates with Neptune using the API key, and initializes a new Neptune run for experiment tracking. The run is configured with the project and entity
    specified in the environment variables.

    Args:
        experiment_name: Optional name of the experiment to log
        run_id: Optional ID of the run to resume

    Returns
    -------
    neptune.Run
        An initialized Neptune run object that can be used to log metrics, artifacts and other experiment data.
        The run will be associated with the specified project.
    """
    api_token = get_neptune_api_token()
    project = parse_project_name()
    run = neptune.init_run(
        api_token=api_token,
        project=project,
        name=experiment_name,
        with_id=run_id,
    )
    return run
