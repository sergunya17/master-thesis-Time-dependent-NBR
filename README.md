# Использование зависимости от времени в задаче рекомендации следующей корзины

## Гиперпараметры

Оптимальные значения указаны для 400 итераций Optuna c оптимизацией метрики Recall@10.


### G-TopFreq (g_top_freq)

#### Описание

- preprocessing: Если `None` - матрица остается в исходном виде, `binary` - все значения больше 1 превращаются в 1, `log` - от каждого значения берется логарифм (``x --> ln(x+1)``).

#### Диапазоны поиска

```python
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary", "log"])
```

#### Оптимальные значения
```python
dunnhumby_best_params = {"preprocessing": None}
instacart_best_params = {"preprocessing": "log"}
tafeng_best_params = {"preprocessing": None}
```


### P-TopFreq (p_top_freq)

#### Описание

- min_freq: Минимальная частота появления айтема. Если айтем встречался меньше, чем ``min_freq`` раз, он не используется.
- preprocessing: Если `None` - матрица остается в исходном виде, `binary` - все значения больше 1 превращаются в 1, `log` - от каждого значения берется логарифм (``x --> ln(x+1)``).

#### Диапазоны поиска

```python
min_freq = trial.suggest_int("min_freq", 1, 20)
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary", "log"])
```

#### Оптимальные значения
```python
dunnhumby_best_params = {"min_freq": 1, "preprocessing": "log"}
instacart_best_params = {"min_freq": 1, "preprocessing": "log"}
tafeng_best_params = {"min_freq": 1, "preprocessing": "log"}
```


### GP-TopFreq (gp_top_freq)

#### Описание

- min_freq: Для матрицы персональной популярности. Минимальная частота появления айтема. Если айтем встречался меньше, чем ``min_freq`` раз, он не используется.
- preprocessing_popular: Для матрицы глобальной популярности. Если `None` - матрица остается в исходном виде, `binary` - все значения больше 1 превращаются в 1, `log` - от каждого значения берется логарифм (``x --> ln(x+1)``).
- preprocessing_personal: Для матрицы персональной популярности. Если `None` - матрица остается в исходном виде, `binary` - все значения больше 1 превращаются в 1, `log` - от каждого значения берется логарифм (``x --> ln(x+1)``).

#### Диапазоны поиска

```python
min_freq = trial.suggest_int("min_freq", 1, 20)
preprocessing_popular = trial.suggest_categorical(
    "preprocessing_popular", [None, "binary", "log"]
)
preprocessing_personal = trial.suggest_categorical(
    "preprocessing_personal", [None, "binary", "log"]
)
```

#### Оптимальные значения
```python
dunnhumby_best_params = {"min_freq": 2, "preprocessing_personal": "log", "preprocessing_popular": None}
instacart_best_params = {"min_freq": 1, "preprocessing_personal": "log", "preprocessing_popular": "binary"}
tafeng_best_params = {"min_freq": 2, "preprocessing_personal": None, "preprocessing_popular": "log"}
```


### ALS (als)

#### Описание

- factors: Размер эмбеддингов.
- regularization: Коэффициент регуляризации.
- implicit: Если `True`, матрица превращается в бинарную (implicit feedback).

#### Диапазоны поиска

```python
factors = trial.suggest_int("factors", 50, 500)
regularization = trial.suggest_float("regularization", 0.001, 0.1, log=True)
implicit = trial.suggest_categorical("implicit", [True, False])
```

#### Оптимальные значения
```python
dunnhumby_best_params = {"factors": 61, "implicit": False, "regularization": 0.021784549326022893}
instacart_best_params = {"factors": 425, "implicit": False, "regularization": 0.09990977570044066}
tafeng_best_params = {"factors": 499, "implicit": False, "regularization": 0.08816574277893643}
```


### UserKNN (user_knn)

#### Описание

- num_nearest_neighbors: Количество ближайших соседей.
- implicit: Если `True`, матрица превращается в бинарную (implicit feedback).

#### Диапазоны поиска

```python
num_nearest_neighbors = trial.suggest_categorical(
    "num_nearest_neighbors", [10, 50, 100, 300, 500, 700, 900, 1100, 1300]
)
implicit = trial.suggest_categorical("implicit", [True, False])
```

#### Оптимальные значения
```python
dunnhumby_best_params = {"implicit": False, "num_nearest_neighbors": 10}
instacart_best_params = {"implicit": False, "num_nearest_neighbors": 10}
tafeng_best_params = {"implicit": False, "num_nearest_neighbors": 10}
```


### UP-CF (up_cf)

#### Описание

- recency: Количество последних корзин, которые считаются актуальными и учитываются моделью.
- q: Параметр локальности для вычисления близости пользователей.
- alpha: Коэффициент асимметрии для косинусного расстояния.
- topk_neighbors: Количество ближайших соседей.
- preprocessing: Если `None` - матрица остается в исходном виде, `binary` - все значения больше 1 превращаются в 1.

#### Диапазоны поиска

```python
recency = trial.suggest_int("recency", 1, 100)
q = trial.suggest_categorical("q", [1, 5, 10, 50, 100, 1000])
alpha = trial.suggest_categorical("alpha", [0, 0.25, 0.5, 0.75, 1])
topk_neighbors = trial.suggest_categorical(
    "topk_neighbors", [None, 10, 100, 300, 500, 700, 900, 1100, 1300]
)
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary"])
```

#### Оптимальные значения
```python
dunnhumby_best_params = {"alpha": 0.75, "preprocessing": None, "q": 5, "recency": 25, "topk_neighbors": 100}
instacart_best_params = {"alpha": 0.5, "preprocessing": None, "q": 5, "recency": 7, "topk_neighbors": 1300}
tafeng_best_params = {"alpha": 0.75, "preprocessing": None, "q": 5, "recency": 76, "topk_neighbors": None}
```


### UP-CF-TA (up_cf_time)

#### Описание

- time_recency: Количество дней до момента покупки последней корзины в трейне, в течение которых корзины считаются актуальными и учитываются моделью.
- q: Параметр локальности для вычисления близости пользователей.
- alpha: Коэффициент асимметрии для косинусного расстояния.
- topk_neighbors: Количество ближайших соседей.
- preprocessing: Если `None` - матрица остается в исходном виде, `binary` - все значения больше 1 превращаются в 1.

#### Диапазоны поиска

```python
time_recency = trial.suggest_int("time_recency", 1, 365)
q = trial.suggest_categorical("q", [1, 5, 10, 50, 100, 1000])
alpha = trial.suggest_categorical("alpha", [0, 0.25, 0.5, 0.75, 1])
topk_neighbors = trial.suggest_categorical(
    "topk_neighbors", [None, 10, 100, 300, 500, 700, 900, 1100, 1300]
)
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary"])
```

#### Оптимальные значения
```python
dunnhumby_best_params = {"alpha": 0.75, "preprocessing": None, "q": 5, "time_recency": 192, "topk_neighbors": 500}
instacart_best_params = {"alpha": 0.25, "preprocessing": None, "q": 5, "time_recency": 64, "topk_neighbors": 1300}
tafeng_best_params = {"alpha": 0.75, "preprocessing": None, "q": 5, "time_recency": 106, "topk_neighbors": 700}
```


### UP-CF-TD (up_cf_time_next_ts)

#### Описание

- time_recency: Количество дней до момента предсказания (или покупки валидационной/тестовой корзины), в течение которых корзины считаются актуальными и учитываются моделью.
- q: Параметр локальности для вычисления близости пользователей.
- alpha: Коэффициент асимметрии для косинусного расстояния.
- topk_neighbors: Количество ближайших соседей.
- preprocessing: Если `None` - матрица остается в исходном виде, `binary` - все значения больше 1 превращаются в 1.

#### Диапазоны поиска

```python
time_recency = trial.suggest_int("time_recency", 1, 365)
q = trial.suggest_categorical("q", [1, 5, 10, 50, 100, 1000])
alpha = trial.suggest_categorical("alpha", [0, 0.25, 0.5, 0.75, 1])
topk_neighbors = trial.suggest_categorical(
    "topk_neighbors", [None, 10, 100, 300, 500, 700, 900, 1100, 1300]
)
preprocessing = trial.suggest_categorical("preprocessing", [None, "binary"])
```

#### Оптимальные значения
```python
dunnhumby_best_params = {"alpha": 0.75, "preprocessing": None, "q": 5, "time_recency": 235, "topk_neighbors": 1100}
instacart_best_params = {"alpha": 0.5, "preprocessing": None, "q": 5, "time_recency": 98, "topk_neighbors": 900}
tafeng_best_params = {"alpha": 0, "preprocessing": None, "q": 1, "time_recency": 310, "topk_neighbors": 100}
```


### TIFU-KNN (tifuknn)

#### Описание

- num_nearest_neighbors: Количество ближайших соседей.
- within_decay_rate: Коэффициент затухания веса корзин внутри группы.
- group_decay_rate: Коэффициент затухания веса групп.
- alpha: Коэффициент баланса между собственным вектором и вектором ближайших соседей.
- group_count: Количество групп у каждого пользователя.

#### Диапазоны поиска

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
group_count = trial.suggest_int("group_count", 2, 23)
```

#### Оптимальные значения
```python
dunnhumby_best_params = {"alpha": 0.4, "group_count": 14, "group_decay_rate": 0.6, "num_nearest_neighbors": 500, "within_decay_rate": 1}
instacart_best_params = {"alpha": 0.8, "group_count": 10, "group_decay_rate": 0.6, "num_nearest_neighbors": 1300, "within_decay_rate": 0.8}
tafeng_best_params = {"alpha": 0.1, "group_count": 17, "group_decay_rate": 0.9, "num_nearest_neighbors": 300, "within_decay_rate": 1}
```


### TIFU-KNN-TA (tifuknn_time_days)

#### Описание

- num_nearest_neighbors: Количество ближайших соседей.
- within_decay_rate: Коэффициент затухания веса корзин внутри группы.
- group_decay_rate: Коэффициент затухания веса групп.
- alpha: Коэффициент баланса между собственным вектором и вектором ближайших соседей.
- group_size_days: Размер групп в днях.
- use_log: Если `True`, используется логарифм в коэффициенте затухания для корзин.

#### Диапазоны поиска

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

#### Оптимальные значения
```python
dunnhumby_best_params = {"alpha": 0.3, "group_decay_rate": 0.2, "group_size_days": 146, "num_nearest_neighbors": 100, "use_log": False, "within_decay_rate": 1}
instacart_best_params = {"alpha": 0.6, "group_decay_rate": 0.4, "group_size_days": 44, "num_nearest_neighbors": 100, "use_log": True, "within_decay_rate": 0.9}
tafeng_best_params = {"alpha": 0.1, "group_decay_rate": 0.4, "group_size_days": 327, "num_nearest_neighbors": 100, "use_log": True, "within_decay_rate": 0.9}
```


### TIFU-KNN-TD (tifuknn_time_days_next_ts)

#### Описание

- num_nearest_neighbors: Количество ближайших соседей.
- within_decay_rate: Коэффициент затухания веса корзин внутри группы.
- group_decay_rate: Коэффициент затухания веса групп.
- alpha: Коэффициент баланса между собственным вектором и вектором ближайших соседей.
- group_size_days: Размер групп в днях.
- use_log: Если `True`, используется логарифм в коэффициенте затухания для корзин.

#### Диапазоны поиска

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

#### Оптимальные значения
```python
dunnhumby_best_params = {"alpha": 0.5, "group_decay_rate": 0.3, "group_size_days": 156, "num_nearest_neighbors": 300, "use_log": False, "within_decay_rate": 1}
instacart_best_params = {"alpha": 0.6, "group_decay_rate": 0.1, "group_size_days": 108, "num_nearest_neighbors": 300, "use_log": True, "within_decay_rate": 0.6}
tafeng_best_params = {"alpha": 0.1, "group_decay_rate": 0.8, "group_size_days": 333, "num_nearest_neighbors": 300, "use_log": True, "within_decay_rate": 0.7}
```
