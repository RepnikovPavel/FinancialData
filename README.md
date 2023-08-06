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
___

```python
AUC 0.7785637114675654
Gini 0.5571274229351308
params = {
    'eta': 0.05,
    'subsample': 0.0,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 3000,
    'gamma':  0.1,
    'subsample': 0.5,
    'colsample_bytree':0.5, 
    'colsample_bylevel':0.5, 
    'colsample_bynode':0.5,
    'lambda': 1.5,
    'alpha': 0.1,
    'scale_pos_weight': np.sum(np.where(Y==0))/np.sum(np.where(Y==1)),
    'min_child_weight': 5,
    'max_delta_step': 5,
    'objective':'binary:logistic',
    'eval_metric': eval_metric, 
    'tree_method':'gpu_hist',
    'gpu_id':0,
    'seed':0
}    
```
add:  
nan augmentation for most valueble categorial feature  
results:  
distribution of answers on train and test are same now
___




# links  
1. [best](https://www.youtube.com/watch?v=NVKDSNM702k)
2. [for begginers](https://www.youtube.com/live/xfKui8OR2dc?feature=share)