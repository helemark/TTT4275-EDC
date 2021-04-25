import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import pandas as pd


"""
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
"""

C = 3
classes = ['Setosa', 'Versicolour', 'Virginica']
features = ['Sepal lenght', 'Sepal width', 'Petal length', 'Petal width']

training_data = []
testing_data = []
tot_vec = []
Tn = [[1, 0, 0]]*30 + [[0, 1, 0] ]*30 + [[0,0,1]]*30
W = np.zeros((3,5))
trenings_terskel = 0.4


len_train_class = 30
len_test_class = 20



def get_data():
    training_data0 = []
    testing_data0 = []
    tot_vec0 = []
    for i in range(C):
        filnavn = ".\Iris_TTT4275\class_"+str(i+1)+".txt"
        with open(filnavn,'r') as myfile:
            data = csv.reader(myfile, delimiter= ',')
            i = 0
            for line in data:
                tot_vec0.append(line)
                if i < len_train_class:
                    training_data0.append(line)
                else:
                    testing_data0.append(line)
                i += 1
   
    for line in training_data0:
        line_new = []
        for value in line:
            line_new.append(float(value))
        line_new.append(1)
        training_data.append(line_new)
        
    for line in tot_vec0:
        line_new = []
        for value in line:
            line_new.append(float(value))
        line_new.append(1)
        tot_vec.append(line_new)
        
        
    for line in testing_data0:
        line_new = []
        for value in line:
            line_new.append(float(value))
        line_new.append(1)
        testing_data.append(line_new)     
    return



   
def plot_petal_data(vec):
    lenght = []
    width = []
    l = int(len(vec)/3)
    for line in vec:
        lenght.append(float(line[2]))
        width.append(float(line[3]))
    plt.plot(lenght[0:l], width[0:l], 'r' 'x')
    plt.plot(lenght[l:2*l], width[l:2*l], 'g' 'x')
    plt.plot(lenght[2*l:3*l], width[2*l:3*l], 'b' 'x')
    return

def plot_sepal_data(vec):
    lenght = []
    width = []
    l = int(len(vec)/3)
    for line in vec:
        lenght.append(float(line[0]))
        width.append(float(line[1]))
    plt.plot(lenght[0:l], width[0:l], 'r' 'x')
    plt.plot(lenght[l:2*l], width[l:2*l], 'g' 'x')
    plt.plot(lenght[2*l:3*l], width[2*l:3*l], 'b' 'x')
    return
        

def averige(data):
    class_1 = [0, 0, 0, 0]
    class_2 = [0, 0, 0, 0]
    class_3 = [0, 0, 0, 0]
    l = int(len(data)/3)
    i = 0
    for i in range(l):
        class_1[0] += data[i][0]/l
        class_1[1] += data[i][1]/l
        class_1[2] += data[i][2]/l
        class_1[3] += data[i][3]/l

        class_2[0] += data[i+l][0]/l
        class_2[1] += data[i+l][1]/l
        class_2[2] += data[i+l][2]/l
        class_2[3] += data[i+l][3]/l

        class_3[0] += data[i+l*2][0]/l
        class_3[1] += data[i+l*2][1]/l
        class_3[2] += data[i+l*2][2]/l
        class_3[3] += data[i+l*2][3]/l
    return class_1, class_2, class_3


def sigmoid(vec):  
    return 1/(1+np.exp(-vec))



def train(vec, alpha):
    global W
    global trenings_terskel
    i = 0
    while(True):
        W_last = W
        n_mse = 0
        for j in range(len(vec)):
            xk = np.matrix(vec[j])
            tk = np.matrix(Tn[j])
            zk = np.dot(W, xk.T).T
            gk = sigmoid(zk)
            mellomleddet = np.multiply((gk-tk),gk)
            d_mse = np.multiply(mellomleddet,(1-gk))
            n_mse += np.dot(d_mse.T, xk)
        
        W = W_last - alpha*n_mse
        #error = np.sum(W-W_last)
        if(np.all(abs(n_mse) <= trenings_terskel)):
            print('Antall iterasjoner:', i+1)
            print('nabla_nmse:',n_mse)
            print('W:', W)
            break
        i += 1
        
def test(vec, number_per_class):
    class_vec = [[], [], []]    
    for i in range(len(vec)):
        xk = np.matrix(vec[i])
        zk = np.dot(W, xk.T).T
        gk = sigmoid(zk[0])
        classified = np.argmax(gk)+1
        if(i<number_per_class):
            class_vec[0].append(classified)
        elif(number_per_class<=i<number_per_class*2):
            class_vec[1].append(classified)
        elif(number_per_class*2<=i):
            class_vec[2].append(classified)
    return class_vec


def error_rate(vec):
    sum_error = 0
    len_class = len(vec[0])
    for i in range(len(vec)):
        sum_error += len_class - vec[i].count(i+1)
    err_t = sum_error/(len(vec)*len_class)
    return err_t


def plot_confusion(vec):
    global classes
    data = {
        classes[0]: [vec[0].count(1), vec[1].count(1), vec[2].count(1)],
        classes[1]: [vec[0].count(2), vec[1].count(2), vec[2].count(2)],
        classes[2]: [vec[0].count(3), vec[1].count(3), vec[2].count(3)]     
        }
    df = pd.DataFrame(data, index = classes)
    print(df)


def plot_feature(vec, index, len_class):
    global C
    class1 = []
    class2 = []
    class3 = []
    for i in range(len_class):
        class1.append(vec[i][index])
        class2.append(vec[i+len_class][index])
        class3.append(vec[i+len_class*2][index])
    plt.hist(class1, color='r', rwidth=1)
    plt.hist(class2, color='g', rwidth=1)
    plt.hist(class3, color='b', rwidth=1)
    plt.title(features[index])
    plt.xlabel('cm')
    plt.ylabel('Nuber')
    plt.show()
    
def remove_feature(vec, index):
    new_vec = []
    for xk in vec:
        xk.pop(index)
        new_vec.append(xk)
    print
    return new_vec
    

if __name__ == "__main__": 
    get_data()
    
    alpha = 0.001
    train(training_data, alpha)

    print('\n\n', 'Testing!')


    print('confusion matrix for the test data')
    test_classified = test(testing_data, len_test_class)
    plot_confusion(test_classified)
    err_t_test = error_rate(test_classified)
    print(err_t_test)
    
    print('\n')
    print('confusion matrix for the train data')
    train_classified = test(training_data, len_train_class)
    plot_confusion(train_classified)
    err_t_train = error_rate(train_classified)
    print(err_t_train)



    #Dette brukes på oppgave 2 a og b, hvor vi fjerner en etter en feature
    #OBS, husk å endre dimensjonene på W
    '''
    training_data_1 = remove_feature(training_data, 1)
    testing_data_1 = remove_feature(testing_data, 1)
    
    training_data_1 = remove_feature(training_data_1, 0)
    testing_data_1 = remove_feature(testing_data_1, 0)
  
    training_data_1 = remove_feature(training_data_1, 0)
    testing_data_1 = remove_feature(testing_data_1, 0)
    
    train(training_data_1, alpha)
    test_classified = test(testing_data_1, len_test_class)
    train_classified = test(training_data_1, len_train_class)

    plot_confusion(test_classified)
    plot_confusion(train_classified)

    err_t_test = error_rate(test_classified)
    print(err_t_test)
    err_t_train = error_rate(train_classified)
    print(err_t_train)
    '''



    #Her plotter vi histogrammer av de ulike featurene for alle klassene.
    #tot_vec = remove_feature(tot_vec, 1)
    #plot_feature(tot_vec, 0, 50)
    #plot_feature(tot_vec, 1, 50)
    #plot_feature(tot_vec, 2, 50)
    #plot_feature(tot_vec, 3, 50)

    
    '''
    #Plotting av ulike features mot hverandre, ikke en del av oppgaen, men for vår egen del.
    f = plt.figure()
    mu_1, mu_2, mu_3 = averige(training_data)
    plot_sepal_data(training_data)
    plot_sepal_data(testing_data)
    class1 = plt.plot(mu_1[0], mu_1[1], 'ro', label = classes[0])
    class2 = plt.plot(mu_2[0], mu_2[1], 'go', label = classes[1])
    class3 = plt.plot(mu_3[0], mu_3[1], 'bo', label = classes[2])
    plt.legend()
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Sepal width [cm]')
    f.savefig("Sepalvssepal.pdf", bbox_inches='tight')
    plt.show()
    
    f = plt.figure()
    plot_petal_data(testing_data)
    plot_petal_data(training_data)
    class1 = plt.plot(mu_1[2], mu_1[3], 'ro', label = classes[0])
    class2 = plt.plot(mu_2[2], mu_2[3], 'go', label = classes[1])
    class3 = plt.plot(mu_3[2], mu_3[3], 'bo', label = classes[2])
    plt.legend()
    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    f.savefig("Petalvspetal.pdf", bbox_inches='tight')
    plt.show()
    '''
