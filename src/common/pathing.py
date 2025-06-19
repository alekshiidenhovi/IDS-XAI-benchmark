import os


def get_local_settings_config_file_path(
    dir_path: str,
    benchmark_id: str,
    file_name: str,
) -> str:
    return os.path.join(
        dir_path,
        benchmark_id,
        file_name,
    )


def get_local_training_config_file_path(
    dir_path: str,
    benchmark_name: str,
    experiment_name: str,
    benchmark_id: str,
    file_name: str,
) -> str:
    return os.path.join(
        dir_path,
        benchmark_name,
        benchmark_id,
        experiment_name,
        file_name,
    )


def get_local_model_file_path(
    dir_path: str,
    benchmark_name: str,
    experiment_name: str,
    benchmark_id: str,
    file_name: str,
) -> str:
    return os.path.join(
        dir_path,
        benchmark_name,
        benchmark_id,
        experiment_name,
        file_name,
    )


def get_local_shap_config_file_path(
    dir_path: str,
    benchmark_name: str,
    experiment_name: str,
    benchmark_id: str,
    file_name: str,
) -> str:
    return os.path.join(
        dir_path,
        benchmark_name,
        benchmark_id,
        experiment_name,
        file_name,
    )


def get_training_time_metrics_file_path(
    dir_path: str,
    benchmark_name: str,
    benchmark_id: str,
) -> str:
    return os.path.join(
        dir_path,
        benchmark_name,
        benchmark_id,
        "training_time_metrics.parquet",
    )


def get_explanation_time_metrics_file_path(
    dir_path: str,
    benchmark_name: str,
    benchmark_id: str,
) -> str:
    return os.path.join(
        dir_path,
        benchmark_name,
        benchmark_id,
        "explanation_time_metrics.parquet",
    )


def get_local_optimization_metrics_file_path(
    dir_path: str,
    benchmark_id: str,
) -> str:
    return os.path.join(
        dir_path,
        benchmark_id,
        "optimization_metrics.json",
    )


def create_dir_if_not_exists(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
