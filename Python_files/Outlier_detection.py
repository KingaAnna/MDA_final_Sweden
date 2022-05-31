#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


# In[ ]:


class my_IsolationForest:
    def __init__(self,contam=0.01,random_state=100):
        self.iso=IsolationForest(contamination=contam)
        self.fips=None
        self.transformed=None
    def __repr__(self):
        return "Isolation Forest for outlier detection"
    def fit(self,X):
        """
        This function fits an Isolation Forest model for outlier detection.
        """
        return self.iso.fit(X)
    def transform(self,X,y=None):
        """
        This function transforms the data by removing the outliers.
        """
        self.fips=self.iso.predict(X)
        mask = self.fips != -1
        self.transformed=X[mask]
        return self.transformed
    def fit_transform(self,X,y=None):
        """
        This function fits an Isolation Forest model and removes the outliers.
        """
        self.fips=self.iso.fit_predict(X)
        mask = self.fips != -1
        self.transformed=X[mask]
        return self.transformed


# In[ ]:


class my_OneClassSVM:
    def __init__(self,contam=0.01):
        self.SVM=OneClassSVM(nu=contam)
        self.fips=None
        self.transformed=None
    def __repr__(self):
        return "Support Vector Machines for outlier detection"
    def fit(self,X):
        """
        This function fits a OneClassSVM model for outlier detection.
        """
        return self.SVM.fit(X)
    def transform(self,X,y=None):
        """
        This function transforms the data by removing the outliers.
        """
        self.fips=self.SVM.predict(X)
        mask = self.fips != -1
        self.transformed=X[mask]
        return self.transformed
    def fit_transform(self,X,y=None):
        """
        This function fits a OneClassSVM model and removes the outliers.
        """
        self.fips=self.SVM.fit_predict(X)
        mask = self.fips != -1
        self.transformed=X[mask]
        return self.transformed

