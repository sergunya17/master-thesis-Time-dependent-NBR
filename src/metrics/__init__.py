from .base import IMetric
from .ndcg import NDCG
from .phr import PHR
from .recall import Recall


METRICS = {
    NDCG.metric_name: NDCG,
    PHR.metric_name: PHR,
    Recall.metric_name: Recall,
}
