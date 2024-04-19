import numpy as np
import core
import extra
import argparse
import time



def KNN_Run_opt(h_param, test_split_ratio=0.2):
    hy_param = extra.hyper_param(h_param)
    f_1_score = []
    accuracy = []
    precision = []
    recall = []
 
    # output = []

    dataset = np.load("data.npy",allow_pickle=True)

    
    for i in hy_param:

        my_knn = core.KNN_Opt(i)
        encoder = i[1]
        if encoder == 'RESNET':
            X_train = dataset[:,1]  # Load the Data
            X_train = np.concatenate([item.flatten() for item in X_train]).reshape(1500, 1024)     #for RESNETs 1st column
            
        elif encoder == 'VIT':
            X_train = dataset[:,2]  # Load the Data
            X_train = np.concatenate([item.flatten() for item in X_train]).reshape(1500, 512)      #for VITs 2nd column
            
        
        
        y_train = dataset[:,3]
        X_train, X_test, y_train, y_test = extra.train_test_split(X_train, y_train, test_size=test_split_ratio, random_state=42)

        my_knn.fit(X_train, y_train)
        predictions = my_knn.predict(X_test)

        f_1_score_t, accuracy_t, precision_t, recall_t = extra.output_of_knn(predictions, y_test)
        f_1_score.append(f_1_score_t)
        accuracy.append(accuracy_t)
        precision.append(precision_t)
        recall.append(recall_t)
        # l_temp = (f_1_score, accuracy, precision, recall)
        # output.append(l_temp)
        #accuracy = extra.accuracy(predictions, y_test)
    return f_1_score, accuracy, precision, recall, len(hy_param)