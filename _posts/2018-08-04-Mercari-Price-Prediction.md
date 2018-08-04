---
tags: [machine learning]
header:   
  image: "/images/bikes.JPG"
excerpt: "How to sugget the price of a product for the seller given their descriptions?"
---
- Task: Build an algorithm that automatically suggests the right product prices.
- Data: User-inputted text descriptions of their products, including details like product category name, brand name, and item condition.


```python
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import pickle
import xgboost
from scipy import sparse
import xgboost

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
```

# Import Data


```python
print ('Importing Data')
current_t = time.time()
train_data = pd.read_table('data/train.tsv')
test_data = pd.read_table('data/test.tsv')
```


```python
print ('Getting features and labels')
current_t = time.time()
def get_feature_label(data):
    # split features and labels
    train_features = data.drop(['price'],axis=1)
    ### log transform
    train_labels =  data.price
    train_labels[train_labels==0]=0.01
    train_labels = np.log(train_labels)
    return train_features,train_labels
train_features,train_labels=get_feature_label(train_data)
nrow_train = train_features.shape[0]
tt_combine = pd.concat([train_features,test_data],axis = 0)

```



## Feature Engineering

### Categorical data


```python
print ('Converting categorical var to numeric')
current_t = time.time()
def category(data):
    cat = data.category_name.str.split('/', expand = True)
    data["main_cat"] = cat[0]
    data["subcat1"] = cat[1]
    data["subcat2"] = cat[2]
    try:
        data["subcat3"] = cat[3]
    except:
        data["subcat3"] = np.nan  
    try:
        data["subcat4"] = cat[4]
    except:
        data["subcat4"] = np.nan  
category(tt_combine)

print ('Handling missing data')   
current_t = time.time()
def missing_data(data, _value = 'None'):
    # Handle missing data
    for col in data.columns:
        data[col].fillna(_value,inplace=True)
missing_data(tt_combine)

print("Coding category data")
le = preprocessing.LabelEncoder()
def cat_to_num(data):
    suf="_le"
    for col in ['brand_name','main_cat','subcat1','subcat2','subcat3','subcat4']:
        data[col+suf] = le.fit_transform(data[col])
        print("{} is transformed to {}".format(col,col+suf))
cat_to_num(tt_combine)
enc = preprocessing.OneHotEncoder()
cat = enc.fit_transform(tt_combine[['main_cat_le','subcat1_le','subcat2_le','subcat3_le','subcat4_le']])


print ('Getting Length of item discription')
tt_combine['Length_of_item_description']=tt_combine['item_description'].apply(len)

print ("Creating numeric Features")
def numeric_to_features(data):
    numeric_features = data[['shipping','item_condition_id','Length_of_item_description','brand_name_le']]
    return numeric_features
numeric_features = numeric_to_features(tt_combine)
print ('Dimension of numeric_features'+str(numeric_features.shape))
print("Categorical data transformed. Time elapsed: " + str(int(time.time()-current_t )) + "s")
```

    Converting categorical var to numeric
    Handling missing data
    Coding category data
    brand_name is transformed to brand_name_le
    main_cat is transformed to main_cat_le
    subcat1 is transformed to subcat1_le
    subcat2 is transformed to subcat2_le
    subcat3 is transformed to subcat3_le
    subcat4 is transformed to subcat4_le
    Getting Length of item discription
    Creating numeric Features
    Dimension of numeric_features(2175894, 4)
    Categorical data transformed. Time elapsed: 33s


### Text Feature


```python
print ("Combining Text")
current_t = time.time()
def text_process(data):
    # Process text    
    # make item_description and name lower case    
    text = list(data.apply(lambda x:'%s %s' %(x['item_description'],x['name']), axis=1))
    return text
text =text_process(tt_combine)
print("Text data combined. Time elapsed: " + str(int(time.time()-current_t )) + "s")


print ('Tfidf')
current_t = time.time()
tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words = 'english',max_features = 5000)
text_features = tfidf.fit_transform(text)
print ('Dimension of text_features'+str(text_features.shape))
print("Tfidf completed. Time elapsed: " + str(int(time.time()-current_t )) + "s")

```

    Combining Text
    Text data combined. Time elapsed: 64s
    Tfidf
    Dimension of text_features(2175894, 5000)
    Tfidf completed. Time elapsed: 1853s



```python
print ("Stacking features")
#  Stacker for sparse data
final_features = sparse.hstack((numeric_features,text_features,cat)).tocsr()
print ('Dimension of final_features'+str(final_features.shape))
train_final_features = final_features[:nrow_train]
test_final_features = final_features[nrow_train:]
print("Data Ready. Time elapsed: " + str(int(time.time()-current_t )) + "s")

```

    Stacking features
    Dimension of final_features(2175894, 6022)
    Data Ready. Time elapsed: 1886s



```python
# save the features
pickle.dump(train_final_features,open('train_features.pkl', "bw"))
pickle.dump(test_final_features,open('test_features.pkl', "bw"))
pickle.dump(train_labels,open('train_labels.pkl', "bw"))
```

## Pick and Tune the Algorithms

An algorithm may be highly sensitive to some of its features. The choose of good parameters may have a dominant effect on the algorithm performance.

In this study, we use GridSearchCV to fine tune the algorithm. I start with default parameters and level it up and down. Based on the GridSearchCV function I will adjust the parameters again. For example, if the GridSearchCV chooses the smallest value for the parameter, I will add a smaller number in the search list.

### Load the data


```python
train_final_features = pickle.load(open('train_features.pkl','br'))
test_final_features = pickle.load(open('test_features.pkl','br'))
train_labels = pickle.load(open('train_labels.pkl','br'))
```

```python
X = (train_final_features)
Y = (train_labels)
current_t = time.time()
xgb = xgboost.XGBRegressor(subsample=0.8,learning_rate=0.5)
param_grid = { "n_estimators" : [300,500,800],
                "max_depth" : [10,15,20], #17-11
                "min_child_weight" : [1,11],
                "gamma":[0,0.2,0.5],
                   }
CV_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid,verbose=1)
X = (train_final_features)
Y = (train_labels)
CV_xgb.fit(X,Y)

print(CV_xgb.best_params_,CV_xgb.best_score_)
print("Modeling complete. Time elapsed: " + str(int(time.time()-current_t)) + "s")
xgb = CV_xgb.best_params_
```

    Initiating grid search
    Fitting 3 folds for each of 54 candidates, totalling 162 fits


### Test

```python
# vectorized error calc
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

# test
def test_reg(reg, features, labels):
    features_train, features_test, labels_train, labels_test = train_test_split(\
                features, labels, test_size=0.8, random_state=0)
    ### fit the classifier using training set, and test on test set
    reg.fit(features_train, (labels_train))
    y_true = labels_test
    y_pred = (reg.predict(features_test))
    y_pred = np.exp(pred_label)
    jag=rmsle(y_true,y_pred)
    print(jag)


test_reg(xgb, train_final_features, train_labels)
```

## Save the results


```python
outfile_name = 'submit.csv'

pred_label = xgb.predict(test_final_features)
pred_label = np.exp(pred_label)
pred_label = pd.DataFrame(np.array(pred_label), columns = ['price'])
pred_label.index.name = 'test_id'
pred_label.to_csv(outfile_name, encoding='utf-8')
```

    Modeling done!
