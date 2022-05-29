#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import warnings
import math
import statsmodels
import numpy as np
from scipy import stats
import statsmodels.api as sm


# In[2]:


class Firth_regression:
    def __init__(self):
        self.beta=[]
        self.waldp=[]
        self.bse=[]
        self.fitll=None
    
    def __repr__(self):
        return "Logistic regression procedure (Firth)."
    
    def fit(self,X,y, start_vec=None, step_limit=1000, convergence_limit=0.0001):
        def firth_likelihood(beta, logit):
            return -(logit.loglike(beta) + 0.5*np.log(np.linalg.det(-logit.hessian(beta))))
        X=sm.add_constant(X)
        logit_model = sm.Logit(y, X)
    
        if start_vec is None:
            start_vec = np.zeros(X.shape[1])
    
        beta_iterations = []
        beta_iterations.append(start_vec)
        for i in range(0, step_limit):
            pi = logit_model.predict(beta_iterations[i])
            W = np.diagflat(np.multiply(pi, 1-pi))
            var_covar_mat = np.linalg.pinv(-logit_model.hessian(beta_iterations[i]))

            # build hat matrix
            rootW = np.sqrt(W)
            H = np.dot(np.transpose(X), np.transpose(rootW))
            H = np.matmul(var_covar_mat, H)
            H = np.matmul(np.dot(rootW, X), H)

            # penalised score
            U = np.matmul(np.transpose(X), y - pi + np.multiply(np.diagonal(H), 0.5 - pi))
            new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U)

            # step halving
            j = 0
            while firth_likelihood(new_beta, logit_model) > firth_likelihood(beta_iterations[i], logit_model):
                new_beta = beta_iterations[i] + 0.5*(new_beta - beta_iterations[i])
                j = j + 1
                if (j > step_limit):
                    sys.stderr.write('Firth regression failed\n')
                    return None

            beta_iterations.append(new_beta)
            if i > 0 and (np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) < convergence_limit):
                break

        return_fit = None
        if np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) >= convergence_limit:
            sys.stderr.write('Firth regression failed\n')
        else:
            # Calculate stats
            fitll = -firth_likelihood(beta_iterations[-1], logit_model)
            intercept = beta_iterations[-1][0]
            beta = beta_iterations[-1][1:].tolist()
            bse = np.sqrt(np.diagonal(np.linalg.pinv(-logit_model.hessian(beta_iterations[-1]))))
        
            self.beta = [intercept] + beta
            self.bse=bse
            self.fitll=fitll
            
            #calculate wald p-values
            self.waldp=[]
            for beta_val, bse_val in zip(self.beta, self.bse):
                self.waldp.append(2 * (1 - stats.norm.cdf(abs(beta_val/bse_val))))   


# In[1]:


class logistic_model:
    def __init__(self):
        self.result=None
    def __repr__(self):
        return "Logistic regression model"
    def fit(self,X,y):
        X=sm.add_constant(X)
        logit_model = sm.Logit(y, X)
        self.result=logit_model.fit()

