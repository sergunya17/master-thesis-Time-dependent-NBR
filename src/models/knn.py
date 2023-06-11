import optuna
import scipy.sparse as sps
from sklearn.neighbors import NearestNeighbors

from src.models.core import IRecommender
from src.dataset import NBRDatasetBase


class UserKNNRecommender(IRecommender):
    def __init__(
        self,
        num_nearest_neighbors: int = 300,
        implicit: bool = False,
    ) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.implicit = implicit

        self._user_vectors = None
        self._nbrs = None

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

        self._nbrs = NearestNeighbors(
            n_neighbors=self.num_nearest_neighbors,
            algorithm="brute",
        ).fit(self._user_vectors)

        return self

    def predict(self, user_ids, topk=None):
        if topk is None:
            topk = self._user_vectors.shape[1]

        user_vectors = self._user_vectors[user_ids, :]

        user_nn_indices = self._nbrs.kneighbors(user_vectors, return_distance=False)

        user_nn_vectors = []
        for nn_indices in user_nn_indices:
            nn_vectors = self._user_vectors[nn_indices, :].mean(axis=0)
            user_nn_vectors.append(sps.csr_matrix(nn_vectors))
        user_nn_vectors = sps.vstack(user_nn_vectors)

        return user_nn_vectors

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        num_nearest_neighbors = trial.suggest_categorical(
            "num_nearest_neighbors", [10, 50, 100, 300, 500, 700, 900, 1100, 1300]
        )
        implicit = trial.suggest_categorical("implicit", [True, False])
        return {
            "num_nearest_neighbors": num_nearest_neighbors,
            "implicit": implicit,
        }
