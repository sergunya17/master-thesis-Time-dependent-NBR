from .als import ALSRecommender
from .core import IRecommender, IRecommenderNextTs
from .knn import UserKNNRecommender
from .statistical import GTopFreqRecommender, GPTopFreqRecommender, PTopFreqRecommender
from .tifuknn import (
    TIFUKNNRecommender,
    TIFUKNNTimeDaysRecommender,
    TIFUKNNTimeDaysNextTsRecommender,
)
from .up_cf import UPCFRecommender, UPCFTimeRecommender, UPCFTimeNextTsRecommender


MODELS = {
    "g_top_freq": GTopFreqRecommender,
    "p_top_freq": PTopFreqRecommender,
    "gp_top_freq": GPTopFreqRecommender,
    "user_knn": UserKNNRecommender,
    "als": ALSRecommender,
    "up_cf": UPCFRecommender,
    "up_cf_time": UPCFTimeRecommender,
    "up_cf_time_next_ts": UPCFTimeNextTsRecommender,
    "tifuknn": TIFUKNNRecommender,
    "tifuknn_time_days": TIFUKNNTimeDaysRecommender,
    "tifuknn_time_days_next_ts": TIFUKNNTimeDaysNextTsRecommender,
}
