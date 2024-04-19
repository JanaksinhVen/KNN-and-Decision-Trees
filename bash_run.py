import numpy as np
import pandas as pd
import core
import extra
import argparse
import sys
#import eval
def run_model(test_data,i):

    dataset = np.load("data.npy",allow_pickle=True)

    my_knn = core.KNN_Opt(i)
    encoder = i[1]
    if encoder == 'RESNET':
        X_train = dataset[:,1]  # Load the Data
        X_test_e= test_data[:,1]
        X_train = np.concatenate([item.flatten() for item in X_train]).reshape(1500, 1024)     #for RESNETs 1st column
        X_test_e= np.concatenate([item.flatten() for item in X_test_e]).reshape(1500, 1024)  

    elif encoder == 'VIT':
        X_train = dataset[:,2]  # Load the Data
        X_test_e= test_data[:,2]
        X_train = np.concatenate([item.flatten() for item in X_train]).reshape(1500, 512)      #for VITs 2nd column
        X_test_e = np.concatenate([item.flatten() for item in X_test_e]).reshape(1500, 512)   
    
    
    y_train = dataset[:,3]
    y_test_e = test_data[:,3]
    #X_train, X_test, y_train, y_test = extra.train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    my_knn.fit(X_train, y_train)
    predictions = my_knn.predict(X_test_e)

    f_1_score, accuracy, precision, recall = extra.output_of_knn(predictions, y_test_e)
    
    return f_1_score, accuracy, precision, recall


if __name__ == "__main__":
    result_list = []
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile",help="write the address of data file") 
   # parser.add_argument("x", type=str ,default = 1.0,help="write the address of data file")    
    args = parser.parse_args()
    
    test_data_p = np.load(args.datafile,allow_pickle=True)
    h_param_p = (3,'VIT','Euclidean')
    f_1_score, accuracy, precision, recall = run_model(test_data_p,h_param_p)
    result_list.append(['VIT',f_1_score, accuracy, precision, recall])
    #sys.stdout.write(str(run_model(test_data_p,h_param_p)))
    h_param_p = (3,'RESNET','Euclidean')
    f_1_score, accuracy, precision, recall = run_model(test_data_p,h_param_p)
    result_list.append(['RESNET',f_1_score, accuracy, precision, recall])
    results = pd.DataFrame(result_list, columns= ['encoder','f-1 score','Accuracy', 'Precision','Recall'])
    print(results)
    #print('f-1 score: ',f_1_score,'Accuracy: ', accuracy,'Precision: ', precision,'Recall: ', recall)
    
    