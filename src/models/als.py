import numpy as np
import optuna
import scipy.sparse as sps
from implicit.cpu.als import AlternatingLeastSquares

from src.models.core import IRecommender
from src.dataset import NBRDatasetBase
from src.settings import RANDOM_SEED


class ALSRecommender(IRecommender):
    def __init__(
        self,
        factors: int = 300,
        regularization: float = 0.01,
        implicit: bool = False,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.regularization = regularization
        self.implicit = implicit

        self._user_vectors = None
        self._als = None

    def fit(self, dataset: NBRDatasetBase):
        user_basket_df = dataset.train_df.copy()
        user_basket_df.reset_index(drop=True, inplace=True)

        df = user_basket_df.explode("basket", ignore_index=True).rename(
            columns={"basket": "item_id"}
        )
        df = df.groupby(["user_id", "item_id"], as_index=False).agg(value=("timestamp", "count"))
        if self.implicit:
            df["value"] = 1.0
        self._user_vectors = sps.csr_matrix(
            (df.value, (df.user_id, df.item_id)),
            shape=(dataset.num_users, dataset.num_items),
        )

        self._als = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            random_state=RANDOM_SEED,
        )
        self._als.fit(self._user_vectors, show_progress=False)

        return self

    def predict(self, user_ids, topk=None):
        if topk is None:
            topk = self._user_vectors.shape[1]

        item_scores = np.dot(self._als.user_factors[user_ids], self._als.item_factors.T)
        not_topk = np.argsort(item_scores, kind="stable")[:, :-topk]
        item_scores[np.arange(item_scores.shape[0])[:, None], not_topk] = 0
        pred_matrix = sps.csr_matrix(item_scores)

        return pred_matrix

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        factors = trial.suggest_int("factors", 50, 500)
        regularization = trial.suggest_float("regularization", 0.001, 0.1, log=True)
        implicit = trial.suggest_categorical("implicit", [True, False])
        return {
            "factors": factors,
            "regularization": regularization,
            "implicit": implicit,
        }
