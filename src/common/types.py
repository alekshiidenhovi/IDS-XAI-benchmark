import typing as T

DATASET_TYPE = T.Literal["training", "testing"]
TRAINING_OBJECTIVE = T.Literal["binary:logistic", "multi:softmax"]
EVAL_METRIC = T.Literal["logloss", "mlogloss"]
