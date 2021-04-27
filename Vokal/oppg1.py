import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd


people = ['m', 'w', 'b', 'g']
count = np.zeros((4,12))
classes = ['ae', 'ah', 'aw', 'eh', 'ei', 'er', 'ih', 'iy','oa', 'oo', 'uh', 'uw']
C = 12

data_list = [[], [], [],  []]

for j in range(4):
    for i in range(12):
        data_list[j].append([])


train_data = []
test_data = []
tot = []
#To get the right dimentions
for i in range(12):
    train_data.append([])
    test_data.append([])
    tot.append([])

#Finds the class of one samples, and returns the index in the claases array
def find_class_index(name):
    vowel = name[3:5]
    index = classes.index(vowel)
    return index

#Finds out weather it is a man, woman boy or girl, and return the corresponding index people
def find_person_index(name):
    person = name[0]
    index = people.index(person)
    return index

#Gets data from the file "vowdata_nohead.dat" and sorts it into test_data and train_data
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


#plots histograms of the features in each class                                         
def plot_hist(peaks, l = None):
    global C
    if l == None:
        l = C
    for i in range(l):
        F1 = []
        F2 = []
        F3 = []
        for value in peaks[i]:
            if(0 not in value):
                F1.append(value[0])
                F2.append(value[1])
                F3.append(value[2])
        f = plt.figure()
        plt.hist(F1)
        plt.hist(F2)
        plt.hist(F3)
        plt.title(classes[i])
        f.savefig(classes[i]+".pdf", bbox_inches='tight')

    return

#Finds the means for all features for each class. Returns a 3x12 array
def find_mean_vec(peaks):
    mean_vec = np.zeros((12, 3))
    l = len(peaks[0])
    for i in range(C):
        for j in range(l):
            mean_vec[i] += peaks[i][j]
    mean_vec /= l
    return mean_vec

#Finds the covariance array for each class. Returns a (3x3)x12 array 
def covariance(peaks, mean_vec):
    sigma = []
    for i in range(C):
        s = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for xk in peaks[i]:
            xk = np.array(xk)
            mu = np.array(mean_vec[i])
            diff = xk - mu
            #print(diff)
            for j in range(len(diff)):
                for k in range(len(diff)):
                    s[j][k] += diff[j]*diff[k]/len(peaks[i])
        sigma.append(s)
    return sigma

#Classifies one sample, and returns the index of                    
def find_class(xk, mu, sigma):
    prob = []
    xk = np.array(xk)    
    for i in range(C):
        mu0 = np.array(mu[i])
        sigma0 = np.array(sigma[i])
        det_sigma0 = np.linalg.det(sigma0)
        inv_sigma0 = np.linalg.inv(sigma0)
        #print('s',inv_sigma0)
        diff = xk - mu0
        #print('d', diff)
        exp1 = np.dot(diff.T, inv_sigma0)
        eksponent = (-1/2)*np.dot(exp1, diff)
        p = np.exp(eksponent)/np.sqrt(((2*np.pi)**3)*det_sigma0)
        prob.append(p)
    #print(np.argmax(prob))
    return np.argmax(prob)
        
#Plots the confusion matrix      
def plot_confusion(vec):
    global classes
    data = {}
    for i in range(C):
        data.update({classes[i]:vec[i]})
    df = pd.DataFrame(data, index = classes)
    print(df)

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

#Takes in a full covariance matrix, and returns a diagonal covariance matrix
def find_dig_cov(vec):
    ide = np.identity(3)
    mat = []
    for i in range(C):
        mat.append(np.multiply(vec[i], ide))
    return np.array(mat) 
    
#Main     
if __name__ == "__main__": 
    get_data()
    #plot_hist(train_data, l = 6)
    #plot_hist(test_data, l = 3)
    plot_hist(tot)
    mean_vec = find_mean_vec(tot)
    print(np.array(mean_vec), '\n')
    sigma = covariance(train_data, mean_vec)
    #print(np.array(sigma))
    
    prob = []
    for j in range(C):
        p = []
        for i in range(69):
            p.append(find_class(test_data[j][i], mean_vec, sigma))
            #print(find_class(test_data[j][i], mean_vec, sigma))
        prob.append([p.count(0), p.count(1), p.count(2), p.count(3), p.count(4), p.count(5), p.count(6), p.count(7), p.count(8),p.count(9), p.count(10), p.count(11)])
    plot_confusion(prob)
    err_t = error_rate(prob)
    print(err_t)


    print('Det')
    dig_sigma = find_dig_cov(sigma)
    #print(dig_sigma)
    prob = []
    for j in range(C):
        p = []
        for i in range(69):
            p.append(find_class(test_data[j][i], mean_vec, dig_sigma))
            #print(find_class(test_data[j][i], mean_vec, sigma))
        prob.append([p.count(0), p.count(1), p.count(2), p.count(3), p.count(4), p.count(5), p.count(6), p.count(7), p.count(8),p.count(9), p.count(10), p.count(11)])
    plot_confusion(prob)
    err_t = error_rate(prob)
    print(err_t)
    
