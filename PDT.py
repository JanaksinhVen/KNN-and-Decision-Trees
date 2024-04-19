# from itertools import chain, combinations
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


def generate_power_set(labels):
    n = len(labels)
    power_set = []
    
    for i in range(2**n):
        subset = []
        for j in range(n):
            if (i >> j) & 1:
                subset.append(labels[j])
        power_set.append(subset)
    
    return power_set

def one_hot_enco(Y, power_set):
    l = len(power_set)
    r = []
    for i in range(len(Y)):
        for j in range(l):
            if power_set[j] == Y[i]:
                r.append(j)
    
    return r
