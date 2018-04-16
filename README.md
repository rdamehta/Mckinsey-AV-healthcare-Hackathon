# Mckinsley-AV-healthcare-Hackathon
Predicting stroke using ensemble and boosting algorithms
Analytics Vidhya and Mckinsley sponsored a 24 hour hackathon with the goal of predicting stroke given patient demographic and certain clinical indicators.

Here is my jupyter notebook for the hackathon. Interestingly I compared a couple of different models. My goal was to see the difference in an ensemble method like random forest and a boosting model from XGBoost.

## Table of Contents
- [Prerequisites](https://github.com/rdamehta/Mckinsley-AV-healthcare-Hackathon/new/master?readme=1#prerequisites)
- [Data cleaning](https://github.com/rdamehta/Mckinsley-AV-healthcare-Hackathon/new/master?readme=1#data-cleaning)
- [Using SMOTE to create synthetic samples and modeling with Random Forest](https://github.com/rdamehta/Mckinsley-AV-healthcare-Hackathon/new/master?readme=1#smote-to-oversample-minority-class-and-random-forest-classifier)
- [SMOTE with svm algorithm to create minority samples, and then random forest classifier](https://github.com/rdamehta/Mckinsley-AV-healthcare-Hackathon/new/master?readme=1#smote-with-svm-algorithm-to-create-minority-samples-and-then-random-forest-classifier)
- [RandomSearchCV the RandomForest Classifier](https://github.com/rdamehta/Mckinsley-AV-healthcare-Hackathon/new/master?readme=1#lets-randomsearchcv-the-randomforest-classifier)
- [Lets use XGBoost](https://github.com/rdamehta/Mckinsley-AV-healthcare-Hackathon/new/master?readme=1#lets-use-xgboost)



### Prerequisites
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
%matplotlib inline
import seaborn as sns
from time import time
from scipy import stats
from scipy.stats import randint as sp_randint
from imblearn.pipeline import make_pipeline, Pipeline
from xgboost import XGBClassifier
```

### Data Cleaning
```python
df = pd.read_csv('train_ajEneEa.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 43400 entries, 0 to 43399
    Data columns (total 12 columns):
    id                   43400 non-null int64
    gender               43400 non-null object
    age                  43400 non-null float64
    hypertension         43400 non-null int64
    heart_disease        43400 non-null int64
    ever_married         43400 non-null object
    work_type            43400 non-null object
    Residence_type       43400 non-null object
    avg_glucose_level    43400 non-null float64
    bmi                  41938 non-null float64
    smoking_status       30108 non-null object
    stroke               43400 non-null int64
    dtypes: float64(3), int64(4), object(5)
    memory usage: 4.0+ MB
    


```python
def clean_data(df):
    
    df.smoking_status.fillna('unknown', inplace = True)
    df.gender = df.gender.map(lambda x: 0 if x == 'Female' else 1)
    df.bmi.fillna(df.bmi.median(), inplace=True)
    df.ever_married= df.ever_married.map(lambda x: 0 if x == 'No' else 1)
    df = pd.get_dummies(df, columns= ['smoking_status','work_type'])
    
    df.Residence_type = df.Residence_type.map(lambda x: 1 if x =='Urban' else 0)
    
    return df

```


```python
df = clean_data(df)
```


```python
df.heart_disease.value_counts()
```




    0    41338
    1     2062
    Name: heart_disease, dtype: int64




```python
df.smoking_status_unknown.value_counts()
```




    0    30108
    1    13292
    Name: smoking_status_unknown, dtype: int64




```python
target = df.pop('stroke')
```


```python
X_train, X_val, y_train, y_val = train_test_split(df.drop(['id'], axis=1), target,
                                                  test_size = .1,
                                                  random_state = 42)

```

### Smote to oversample minority class, and random forest classifier


```python
#regular smote, and rf this will give me a score of .759
sm = SMOTE(random_state=12, ratio = 'minority', n_jobs=-1)#try different algorithms (borderline1, borderline 2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)


rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train_res, y_train_res)


probs = rf.predict_proba(X_val)
predictions = rf.predict(X_val)

roc_auc_score(y_val, probs[:,1]), roc_auc_score(y_val, predictions)
```


```python
test = pd.read_csv('test_v2akXPA.csv')
test = clean_data(test)

test_probas = rf.predict_proba(test.drop(['id'], axis=1))
```


```python
submission = pd.DataFrame({'id':test.id,
              'stroke':test_probas[:,1]})
submission.to_csv('probas_submission.csv', index=False, index_label=None)

```


```python

```

### SMOTE with svm algorithm to create minority samples, and then random forest classifier


```python
sm = SMOTE(ratio='minority', kind='svm', n_jobs=-1)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
```


```python
## with svm smote algorithm
X_train_borderline, y_train_borderline = sm.fit_sample(X_train, y_train)#variable names need to me changed

rf_borderline = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf_borderline.fit(X_train_borderline, y_train_borderline)

probs = rf_borderline.predict_proba(X_val)
roc_auc_score(y_val, probs[:,1])

#roc_auc_score(y_val, probs[:,0])
```




    0.7478183081043537




```python
test = pd.read_csv('test_v2akXPA.csv')
test = clean_data(test)

test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status_formerly smoked</th>
      <th>smoking_status_never smoked</th>
      <th>smoking_status_smokes</th>
      <th>smoking_status_unknown</th>
      <th>work_type_Govt_job</th>
      <th>work_type_Never_worked</th>
      <th>work_type_Private</th>
      <th>work_type_Self-employed</th>
      <th>work_type_children</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36306</td>
      <td>1</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>83.84</td>
      <td>21.1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>61829</td>
      <td>0</td>
      <td>74.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>179.50</td>
      <td>26.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14152</td>
      <td>0</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>95.16</td>
      <td>21.2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12997</td>
      <td>1</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>94.76</td>
      <td>23.4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40801</td>
      <td>0</td>
      <td>63.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>83.57</td>
      <td>27.6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_probas = rf_borderline.predict_proba(test.drop(['id'], axis=1))

rf_svm_predictions = pd.DataFrame({'id':test.id,
                         'stroke':test_probas[:,1]})

rf_svm_predictions.to_csv('smote_svm_submission.csv', index=False, index_label=None)
```

### Lets RandomSearchCv the RandomForest Classifier


```python
clf_rf = RandomForestClassifier(n_jobs=-1)
smote_enn = SMOTEENN(smote = sm)#we created an sm object with svm algorithm
pipeline = Pipeline([('smote_enn', smote_enn),
                     ('clf_rf', clf_rf)])

# specify parameters and distributions to sample from
param_dist = {"clf_rf__n_estimators": sp_randint(10,1000),
              "clf_rf__max_depth": [3, None],
              "clf_rf__max_features": sp_randint(1, 11),
              "clf_rf__min_samples_split": sp_randint(2, 11),
              "clf_rf__min_samples_leaf": sp_randint(1, 11),
              "clf_rf__bootstrap": [True, False],
              "clf_rf__criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(pipeline, param_distributions = param_dist,
                                   n_iter = n_iter_search,
                                   scoring = 'roc_auc' )

start = time()
random_search.fit(df.drop(['id'], axis = 1), target)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
#report(random_search.cv_results_)
```

    RandomizedSearchCV took 1474.22 seconds for 20 candidates parameter settings.
    


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-31-8a529e9a29f9> in <module>()
         23 print("RandomizedSearchCV took %.2f seconds for %d candidates"
         24       " parameter settings." % ((time() - start), n_iter_search))
    ---> 25 report(random_search.cv_results_)
    

    NameError: name 'report' is not defined



```python
#random_search.cv_results_
```


```python
random_search.best_params_
```




    {'clf_rf__bootstrap': True,
     'clf_rf__criterion': 'entropy',
     'clf_rf__max_depth': 3,
     'clf_rf__max_features': 9,
     'clf_rf__min_samples_leaf': 4,
     'clf_rf__min_samples_split': 3,
     'clf_rf__n_estimators': 829}




```python
test_probas = random_search.predict_proba(test.drop(['id'], axis=1))

rs_predictions = pd.DataFrame({'id':test.id,
                         'stroke':test_probas[:,1]})

rs_predictions.to_csv('randomsearchcv_submission.csv', index=False, index_label=None)
```


```python
rs_predictions.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36306</td>
      <td>0.681182</td>
    </tr>
    <tr>
      <th>1</th>
      <td>61829</td>
      <td>0.774926</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14152</td>
      <td>0.004214</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12997</td>
      <td>0.005095</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40801</td>
      <td>0.301565</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pickle
with open('randomsearchcv.pickle', 'wb') as handle:
    pickle.dump(random_search, handle, protocol=pickle.HIGHEST_PROTOCOL)
```


```python

```

### Lets use XGBoost


```python
from xgboost import XGBClassifier
```


```python
X_val.values
```




    array([[ 0., 54.,  0., ...,  1.,  0.,  0.],
           [ 0., 19.,  0., ...,  1.,  0.,  0.],
           [ 0., 27.,  0., ...,  1.,  0.,  0.],
           ...,
           [ 1.,  5.,  0., ...,  0.,  0.,  1.],
           [ 1., 34.,  0., ...,  1.,  0.,  0.],
           [ 0., 48.,  0., ...,  1.,  0.,  0.]])




```python
#BEST SCORE SO FAR
xgb_clf = XGBClassifier(n_jobs = -1, )
xgb_clf.fit(X_train_borderline, y_train_borderline)
probas = xgb_clf.predict_proba(X_val.values)
roc_auc_score(y_val.values, probas[:,1])
```




    0.8271470307639504




```python
test_values = test.drop(['id'], axis=1).values
test_probas = xgb_clf.predict_proba(test_values)

xgb_predictions = pd.DataFrame({'id':test.id,
                         'stroke':test_probas[:,1]})

xgb_predictions.to_csv('xgboost_submission.csv', index=False, index_label=None)
```


```python
np.linspace(.5,.9,10)
```




    array([0.5       , 0.54444444, 0.58888889, 0.63333333, 0.67777778,
           0.72222222, 0.76666667, 0.81111111, 0.85555556, 0.9       ])




```python
#lets put this in a pipeline.

# pipeline = Pipeline([('smote_enn', smote_enn),
#                      ('xgb_clf', xgb_clf)])

# specify parameters and distributions to sample from
# sm = SMOTE(ratio='minority', kind='svm', n_jobs=-1)
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

xgb_clf = XGBClassifier(n_jobs = -1)

param_dist = {"n_estimators": sp_randint(100,1000),
              "max_depth": sp_randint(3, 10),
              "learning_rate": stats.uniform(0.01, 0.6),
              "colsample_bytree": np.linspace(.5,.9,10),
              "min_child_weight": sp_randint(1,6) 
             }

n_iter_search = 10
random_search = RandomizedSearchCV(xgb_clf, param_distributions = param_dist,
                                   n_iter = n_iter_search,
                                   scoring = 'roc_auc',
                                   n_jobs=1)

start = time()
random_search.fit(X_train_res, y_train_res)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
```

    RandomizedSearchCV took 527.52 seconds for 10 candidates parameter settings.
    


```python
random_search.best_params_
```




    {'colsample_bytree': 0.6777777777777778,
     'learning_rate': 0.28090313429809166,
     'max_depth': 9,
     'min_child_weight': 2,
     'n_estimators': 271}




```python
test_values = test.drop(['id'], axis=1).values
test_probas = random_search.predict_proba(test_values)

xgb_predictions = pd.DataFrame({'id':test.id,
                         'stroke':test_probas[:,1]})

xgb_predictions.to_csv('xgbmedian_submission.csv', index=False, index_label=None)
```


```python
#This gave me a score of .846 which put me in the top 10% of the public leaderboard
```


```python

```
