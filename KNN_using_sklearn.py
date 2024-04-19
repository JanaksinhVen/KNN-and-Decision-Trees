import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import extra

def KNN_sklearn(h_param,test_size_p=0.2):
    hy_param = extra.hyper_param(h_param)
    
    dataset = np.load("data.npy",allow_pickle=True)

    for i in hy_param:
        encoder = i[1]
        if encoder == 'RESNET':
            X_train = dataset[:,1]  # Load the Data
            X_train = np.concatenate([item.flatten() for item in X_train]).reshape(1500, 1024)     #for RESNETs 1st column
            
        elif encoder == 'VIT':
            X_train = dataset[:,2]  # Load the Data
            X_train = np.concatenate([item.flatten() for item in X_train]).reshape(1500, 512)      #for VITs 2nd column
            
        
        
        Y_train = dataset[:,3]

        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=test_size_p, random_state=22)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)
        #print(f"Accuracy: {accuracy:.2f}")

    return accuracy