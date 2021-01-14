#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

TITANIC_PATH = '/cxldata/datasets/project/titanic'


# In[5]:


import pandas as pd
def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


# In[7]:


train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")


# In[12]:


train_data.head()


# In[13]:


train_data.info()


# In[14]:


train_data.describe()


# In[15]:


train_data["Sex"].value_counts()[1]


# In[20]:


from sklearn.base import BaseEstimator,TransformerMixin


# In[21]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# In[24]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])


# In[25]:


num_pipeline.fit_transform(train_data)


# In[30]:


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# In[33]:


from sklearn.preprocessing import OneHotEncoder


# In[34]:


cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])


# In[35]:


cat_pipeline.fit_transform(train_data)


# In[38]:


from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[39]:


X_train = preprocess_pipeline.fit_transform(train_data)


# In[40]:


y_train = train_data["Survived"]


# In[44]:


from sklearn.svm import SVC


# In[45]:


svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train, y_train)


# In[51]:


X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)


# In[54]:


from sklearn.model_selection import cross_val_score


# In[55]:


svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()


# In[59]:


from sklearn.ensemble import RandomForestClassifier


# In[60]:


forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()

