Есть достаточно большой дата-сет, многолетний, по займам. 

Вам нужно предсказать бинарный таргет.

 

Оценка модели будет делаться по метрике Gini:

$$
Gini = 2 * AUC - 1 \text{ где AUC - это ROC AUC}
$$

Для построения модели, предоставляется обучающий набор клиентов X_train.csv с информацией о заемщиках. Целевая переменная находится в файле y_train.csv (1 = "невозврат")

Необходимо для каждого клиента из тестовой выборки test.csv предсказать loan_status, который равен 1, если займ не возвращен и 0, если заемщик вернул займ.

Результат должен быть представлен в виде упакованного в zip CSV-файла с названием answer.csv и с колонками index и loan_status (где проставлена вероятность значения 1)

```python 
AUC 0.723967030044554
Gini 0.44793406008910797
params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100,
            'eval_metric': 'auc',
            'tree_method':'gpu_hist',
            'gpu_id':0,
            'seed':0
        }    
```





# links  
1. [best](https://www.youtube.com/watch?v=NVKDSNM702k)
2. [for begginers](https://www.youtube.com/live/xfKui8OR2dc?feature=share)