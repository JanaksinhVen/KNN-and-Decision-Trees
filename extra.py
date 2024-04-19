import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_test_split(X, y, test_size=0.2, random_state=None):
    #print('test  ------ ', X.shape, y.shape)
    if random_state is not None:
        np.random.seed(random_state)
    
    num_samples = X.shape[0]
    #print(num_samples)
    num_test_samples = int(num_samples * test_size)
    shuffled_indices = np.random.permutation(num_samples)
    #print('num samples - ', X.shape)
    #print(shuffled_indices.shape)
    #print('x and y = ', X.shape, '  --  ', y.shape)
    
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    X_train = X_shuffled[:-num_test_samples]
    y_train = y_shuffled[:-num_test_samples]
    X_test = X_shuffled[-num_test_samples:]
    y_test = y_shuffled[-num_test_samples:]
    
    return X_train, X_test, y_train, y_test


def accuracy(y_pred, y_true):
    correct_predictions = np.sum(y_pred == y_true)
    total_predictions = len(y_pred)
    accuracy = correct_predictions / total_predictions
    return accuracy

def hyper_param(h_param):

    h_list = list(product(*h_param.values()))
    return h_list

def output_of_knn(predictions_p, y_test_p):
    

    accuracy = accuracy_score(y_test_p, predictions_p)
    precision_macro = precision_score(y_test_p, predictions_p, average='macro',zero_division=0)
    recall_macro = recall_score(y_test_p, predictions_p, average='macro',zero_division=0)
    f1_macro = f1_score(y_test_p, predictions_p, average='macro')
    return f1_macro, accuracy,precision_macro, recall_macro