import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture


people = ['m', 'w', 'b', 'g']
count = np.zeros((4,12))
classes = ['ae', 'ah', 'aw', 'eh', 'ei', 'er', 'ih', 'iy','oa', 'oo', 'uh', 'uw']
C = 12


#List used as a middleman to sort the data from the file
data_list = [[], [], [],  []]

#To get the right dimentions on data_list
for j in range(4):
    for i in range(12):
        data_list[j].append([])

#Sorts data lists
train_data = []
test_data = []
tot = []
for i in range(12):
    train_data.append([])
    test_data.append([])
    tot.append([])

#Finds the index in "classes" for a goven vowel
def find_class_index(name):
    vowel = name[3:5]
    index = classes.index(vowel)
    return index

#Finds the index in "persons" for a givem person
def find_person_index(name):
    person = name[0]
    index = people.index(person)
    return index

#Gets and sorts the data form the datafile
def get_data():
    global data_list
    filnavn = "vowdata_nohead.dat"
    with open(filnavn,'r') as myfile:
        data = csv.reader(myfile, delimiter= ' ')
        for line in data:
            while('' in line):
                line.remove('')
            person = find_person_index(line[0])
            index = find_class_index(line[0])
            xk = [float(line[10]), float(line[11]), float(line[12])]
            data_list[person][index].append(xk)

    for i in range(4):
        for j in range(C):
            l = len(data_list[i][j])
            train_data[j]+=data_list[i][j][:round(l/2)]
            test_data[j]+=data_list[i][j][round(l/2):]
            tot[j]+=data_list[i][j]


#Takes all the training data and gets the GMM information
def train(vec, n):
    global C
    gm_vec = []
    for i in range(C):
        F = np.array(vec[i])
        gm = GaussianMixture(n_components=n, random_state=0).fit(F)
        gm_vec.append(gm)
    return gm_vec

#Implementation of the function calculating the probability of one sample being in one class.
def gauss(xk, sigma0, mu0):
    det_sigma0 = np.linalg.det(sigma0)
    inv_sigma0 = np.linalg.inv(sigma0)
    diff = xk - mu0
    exp1 = np.dot(diff[0].T, inv_sigma0)
    eksponent = (-1/2)*np.dot(exp1, diff[0])
    p = np.exp(eksponent)/np.sqrt(((2*np.pi)**3)*det_sigma0)
    return p

#Plots the confusion matrix
def plot_confusion(vec):
    global classes
    data = {}
    for i in range(C):
        data.update({classes[i]:vec[i]})
    df = pd.DataFrame(data, index = classes)
    print(df)
    
#Uses the gauss function to calculate the probability of one sample being in each class 
def find_class(xk, gm_vec):
    xk = np.array(xk)
    prob =[]
    for gm in gm_vec:
        mu = gm.means_
        c = gm.weights_
        sigma = find_dig_cov(gm.covariances_)      
        p = 0
        for i in range(len(mu)):
            mu0 = np.array(mu[i])
            sigma0 = np.array(sigma[i])
            p += gauss(xk, sigma0, mu0)*c[i]
        prob.append(p)
    return np.argmax(prob)

#Finds the error rate    
def error_rate(vec):
    sum_not_error = 0
    tot = 0
    len_class = len(vec[0])
    for i in range(len(vec)):
        sum_not_error += vec[i][i]
        tot += sum(vec[i])
    err_t = (tot-sum_not_error)/tot
    return err_t

#Takes in a 3x3 covariance matrix and returns the diagonal version of it
def find_dig_cov(vec):
    ide = np.identity(3)
    return np.multiply(vec, ide)

#Main
if __name__ == "__main__":
    get_data()
    gm_vec = train(train_data, 2)
    prob = []
    for j in range(C):
        p = []
        for i in range(69):
            p.append(find_class([test_data[j][i]], gm_vec))
        prob.append([p.count(0), p.count(1), p.count(2), p.count(3), p.count(4), p.count(5), p.count(6), p.count(7), p.count(8),p.count(9), p.count(10), p.count(11)])
    plot_confusion(prob)
    err_t = error_rate(prob)
    print(err_t)
   
