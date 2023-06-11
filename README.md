# Использование зависимости от времени в задаче рекомендации следующей корзины

## Диапазоны поиска гиперпараметров

### G-TopFreq (g_top_freq)
```python
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary", "log"])
```

### P-TopFreq (p_top_freq)
```python
min_freq = trial.suggest_int("min_freq", 1, 20)
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary", "log"])
```

### GP-TopFreq (gp_top_freq)
```python
min_freq = trial.suggest_int("min_freq", 1, 20)
preprocessing_popular = trial.suggest_categorical(
    "preprocessing_popular", [None, "binary", "log"]
)
preprocessing_personal = trial.suggest_categorical(
    "preprocessing_personal", [None, "binary", "log"]
)
```

### ALS (als)
```python
factors = trial.suggest_int("factors", 50, 500)
regularization = trial.suggest_float("regularization", 0.001, 0.1, log=True)
implicit = trial.suggest_categorical("implicit", [True, False])
```

### UserKNN (user_knn)
```python
num_nearest_neighbors = trial.suggest_categorical(
    "num_nearest_neighbors", [10, 50, 100, 300, 500, 700, 900, 1100, 1300]
)
implicit = trial.suggest_categorical("implicit", [True, False])
```

### UP-CF (up_cf)
```python
recency = trial.suggest_int("recency", 1, 100)
q = trial.suggest_categorical("q", [1, 5, 10, 50, 100, 1000])
alpha = trial.suggest_categorical("alpha", [0, 0.25, 0.5, 0.75, 1])
topk_neighbors = trial.suggest_categorical(
    "topk_neighbors", [None, 10, 100, 300, 500, 700, 900, 1100, 1300]
)
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary"])
```

### UP-CF-TA (up_cf_time)
```python
q = trial.suggest_categorical("q", [1, 5, 10, 50, 100, 1000])
alpha = trial.suggest_categorical("alpha", [0, 0.25, 0.5, 0.75, 1])
topk_neighbors = trial.suggest_categorical(
    "topk_neighbors", [None, 10, 100, 300, 500, 700, 900, 1100, 1300]
)
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary"])
time_recency = trial.suggest_int("time_recency", 1, 365)
```

### UP-CF-TD (up_cf_time_next_ts)
```python
q = trial.suggest_categorical("q", [1, 5, 10, 50, 100, 1000])
alpha = trial.suggest_categorical("alpha", [0, 0.25, 0.5, 0.75, 1])
topk_neighbors = trial.suggest_categorical(
    "topk_neighbors", [None, 10, 100, 300, 500, 700, 900, 1100, 1300]
)
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary"])
time_recency = trial.suggest_int("time_recency", 1, 365)
```


### TIFU-KNN (tifuknn)
```python
num_nearest_neighbors = trial.suggest_categorical(
    "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
)
within_decay_rate = trial.suggest_categorical(
    "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
group_decay_rate = trial.suggest_categorical(
    "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
alpha = trial.suggest_categorical(
    "alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
```

### TIFU-KNN-TA (tifuknn_time_days)
```python
num_nearest_neighbors = trial.suggest_categorical(
    "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
)
within_decay_rate = trial.suggest_categorical(
    "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
group_decay_rate = trial.suggest_categorical(
    "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
alpha = trial.suggest_categorical(
    "alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
group_size_days = trial.suggest_int("group_size_days", 1, 365)
use_log = trial.suggest_categorical("use_log", [True, False])
```

### TIFU-KNN-TD (tifuknn_time_days_next_ts)
```python
num_nearest_neighbors = trial.suggest_categorical(
    "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
)
within_decay_rate = trial.suggest_categorical(
    "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
group_decay_rate = trial.suggest_categorical(
    "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
alpha = trial.suggest_categorical(
    "alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)
group_size_days = trial.suggest_int("group_size_days", 1, 365)
use_log = trial.suggest_categorical("use_log", [True, False])
```
