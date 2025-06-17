from common.config import TrainingConfig


def get_experiment_group_name(
    current_datetime: str,
    training_config: TrainingConfig,
):
    experiment_group_name = f"{current_datetime}-xgb_train-{training_config.n_estimators}_noftrees-{training_config.max_depth}_depth-{training_config.learning_rate:.3f}_lr-{training_config.subsample:.3f}_subsample-{training_config.colsample_bytree:.3f}_colsample"
    return experiment_group_name
