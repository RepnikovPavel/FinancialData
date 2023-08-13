Есть достаточно большой дата-сет, многолетний, по займам. 

Вам нужно предсказать бинарный таргет.

 

Оценка модели будет делаться по метрике Gini:

$$
Gini = 2 * AUC - 1 \text{ где AUC - это ROC AUC}
$$

Для построения модели, предоставляется обучающий набор клиентов X_train.csv с информацией о заемщиках. Целевая переменная находится в файле y_train.csv (1 = "невозврат")

Необходимо для каждого клиента из тестовой выборки test.csv предсказать loan_status, который равен 1, если займ не возвращен и 0, если заемщик вернул займ.

Результат должен быть представлен в виде упакованного в zip CSV-файла с названием answer.csv и с колонками index и loan_status (где проставлена вероятность значения 1)

___  

```python3 
add:  
nan augmentation for most valueble categorial feature  
results:  
distribution of answers on train and test are same now
```

___  

```python
add: split dataset by time and train a few models 
add: one hot encoding for a few categorial data
add: make a new categorial feautures from pairs of features
```


# links  
1. [best](https://www.youtube.com/watch?v=NVKDSNM702k)
2. [for begginers](https://www.youtube.com/live/xfKui8OR2dc?feature=share)
3. [about xgboost from kagglers](https://www.kaggle.com/code/bextuychiev/20-burning-xgboost-faqs-answered-to-use-like-a-pro)
