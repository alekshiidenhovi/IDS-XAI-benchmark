from datasets.unsw import UNSW_NB15
from common.types import DATASET_TYPE


def prepare_dataset(
    target_column_name: str,
    dataset_type: DATASET_TYPE,
    val_proportion: float,
    random_state: int,
    dataset_shuffle: bool,
):
    train_dataset = UNSW_NB15(dataset_type=dataset_type)
    train_feature_matrix = train_dataset.get_feature_matrix()
    train_target_series = train_dataset.get_target_series(
        target_column_name=target_column_name
    )
    X_train, X_validation, y_train, y_validation = (
        train_dataset.create_train_val_splits(
            train_feature_matrix,
            train_target_series,
            val_proportion=val_proportion,
            random_state=random_state,
            dataset_shuffle=dataset_shuffle,
        )
    )
    return X_train, X_validation, y_train, y_validation
