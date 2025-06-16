import wandb
import os
import typing as T
from dotenv import load_dotenv


def init_wandb_api_client():
    """Initialize a W&B API client."""
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    api = wandb.Api(api_key=WANDB_API_KEY)
    return api


def init_wandb_run(
    experiment_name: T.Optional[str] = None,
    experiment_group_name: T.Optional[str] = None,
    reinit: bool = False,
):
    """
    Initialize and configure a Weights & Biases logger.

    This function loads environment variables from a .env file, authenticates with W&B using the API key,
    and initializes a new W&B run for experiment tracking. The run is configured with the project and entity
    specified in the environment variables.

    Args:
        experiment_name: Optional name of the experiment to log
        experiment_group_name: Optional name of the experiment group to log
        reinit: Whether to reinitialize the run if it already exists

    Returns
    -------
    wandb.wandb_run.Run
        An initialized W&B run object that can be used to log metrics, artifacts and other experiment data.
        The run will be associated with the specified project and entity.

    Notes
    -----
    Required environment variables in .env file:
    - WANDB_API_KEY: The Weights & Biases API key for authentication
    - WANDB_PROJECT: The name of the W&B project to log experiments to
    - WANDB_ENTITY: The W&B username or team name that owns the project
    """
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")

    wandb.login(key=WANDB_API_KEY)

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=experiment_group_name,
        job_type=experiment_name,
        name=experiment_name,
        reinit=reinit,
    )

    return run
