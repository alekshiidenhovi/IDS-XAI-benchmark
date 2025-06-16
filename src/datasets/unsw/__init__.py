import pandas as pd
import kagglehub
import typing as T
from common.types import DATASET_TYPE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class UNSW_NB15:
    def __init__(self, dataset_type: DATASET_TYPE):
        self.encoder_dict: T.Dict[str, LabelEncoder] = {}
        self.categorical_features: T.List[str] = []
        self.df: pd.DataFrame = self._load_unsw_nb15(dataset_type=dataset_type)
        self._preprocess_features()

    def _load_unsw_nb15(self, dataset_type: DATASET_TYPE) -> pd.DataFrame:
        """
        Load the UNSW-nb15 dataset from Kaggle.
        """
        df = kagglehub.dataset_load(
            adapter=kagglehub.KaggleDatasetAdapter.PANDAS,
            handle="dhoogla/unswnb15",
            path=f"UNSW_NB15_{dataset_type}-set.parquet",
        )
        return df

    def _rename_features(self) -> None:
        """
        Rename the features of the UNSW-nb15 dataset.
        """
        self.df.rename(columns=renamed_features_dict, inplace=True)

    def _encode_categorical_features(self) -> None:
        """
        Encode the categorical features of the UNSW-nb15 dataset. Stores the encoders in the encoder_dict.
        """
        for feature in categorical_features:
            encoder = LabelEncoder()
            self.df[f"{feature}"] = encoder.fit_transform(self.df[feature])
            self.encoder_dict[feature] = encoder

    def _preprocess_features(self) -> None:
        """
        Preprocess the UNSW-nb15 dataset.
        """
        self._rename_features()
        self._encode_categorical_features()

    def get_feature_matrix(self) -> pd.DataFrame:
        """
        Parse the feature matrix of the UNSW-nb15 dataset.
        """
        feature_matrix = self.df.drop(columns=target_columns)
        return feature_matrix

    def get_target_series(self, target_column_name: str) -> pd.Series:
        """
        Parse the target matrix of the UNSW-nb15 dataset.
        """
        target_series = self.df[target_column_name]
        return target_series

    def create_train_val_splits(
        self,
        feature_matrix: pd.DataFrame,
        target_series: pd.Series,
        val_size: float,
        random_state: int,
        dataset_shuffle: bool,
    ) -> T.Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_validation, y_train, y_validation = train_test_split(
            feature_matrix,
            target_series,
            test_size=val_size,
            stratify=target_series,
            random_state=random_state,
            shuffle=dataset_shuffle,
        )
        return X_train, X_validation, y_train, y_validation


renamed_features_dict: T.Dict[str, str] = {
    "proto": "transaction_protocol",
    "state": "state",
    "dur": "duration",
    "sbytes": "src_to_dest_bytes",
    "dbytes": "dest_to_src_bytes",
    "sloss": "src_loss",
    "dloss": "dest_loss",
    "service": "protocol_name",
    "sload": "src_bits_per_second",
    "dload": "dest_bits_per_second",
    "spkts": "src_to_dest_packets",
    "dpkts": "dest_to_src_packets",
    "swin": "src_tcp_window",
    "dwin": "dest_tcp_window",
    "stcpb": "src_tcp_seq_number",
    "dtcpb": "dest_tcp_seq_number",
    "smean": "src_mean_packet_size",
    "dmean": "dest_mean_packet_size",
    "trans_depth": "transaction_depth",
    "response_body_len": "resp_body_length",
    "sjit": "src_jitter",
    "djit": "dest_jitter",
    "sinpkt": "src_inter_packet_arrival_time",
    "dinpkt": "dest_inter_packet_arrival_time",
    "tcprtt": "sum_synack_ackdat",
    "synack": "syn_to_synack_time",
    "ackdat": "synack_to_ack_time",
    "is_sm_ips_ports": "src_equals_dest",
    "ct_flw_http_mthd": "http_methods_count",
    "is_ftp_login": "is_ftp_login",
    "ct_ftp_cmd": "ftp_session_count",
    "ct_src_dport_ltm": "same_src_addr_dest_port_count",
    "ct_dst_sport_ltm": "same_dest_addr_src_port_count",
    "rate": "rate",
    "attack_cat": "attack_category",
    "label": "is_attack",
}

categorical_features: T.List[str] = [
    "transaction_protocol",
    "protocol_name",
    "state",
    "attack_category",
]

binary_target_column: str = "is_attack"
multilabel_target_column: str = "attack_category"
target_columns: T.List[str] = [binary_target_column, multilabel_target_column]
