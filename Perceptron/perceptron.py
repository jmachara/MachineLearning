import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import random as rnd
class perceptron:
    def standard_perceptron(r):
        data = pd.read_csv('./Data/train.csv')
        w = np.zeros(4)
        for t in range(0,10):
            rand_data = data.sample(frac=1).to_numpy()
            for i in range(0,len(rand_data)):
                x = rand_data[i,0:4]
                y = rand_data[i,4]
                if y == 0:
                    y = -1
                prediction = x@w
                if prediction*y <= 0:
                    for j in range(0,len(w)):
                        w[j] += y*x[j]*r
        return w
    def voted_perceptron(r):
        C = {}
        w_dict = {}
        m = 0
        data = pd.read_csv('./Data/train.csv').to_numpy()
        w = np.zeros(4)
        for t in range(0,10):
            for i in range(0,len(data)):
                x = data[i,0:4]
                y = data[i,4]
                if y == 0:
                    y = -1
                prediction = x@w
                if prediction*y <= 0:
                    for j in range(0,len(w)):
                        w[j] += y*x[j]*r
                    m += 1
                    C[m] = 1
                    w_dict[m] = w.copy()
                else:
                    C[m] += 1
        return w_dict,C,m
    def average_perceptron(r):
        data = pd.read_csv('./Data/train.csv').to_numpy()
        w = np.zeros(4)
        a = np.zeros(4)
        for t in range(0,10):
            for i in range(0,len(data)):
                x = data[i,0:4]
                y = data[i,4]
                if y == 0:
                    y = -1
                prediction = x@w
                if prediction*y <= 0:
                    for j in range(0,len(w)):
                        w[j] += y*x[j]*r
                a = np.add(a,w)
        return a      
    def get_perceptron_error(w):
        error_count = 0
        test_data = pd.read_csv('./Data/test.csv').to_numpy()
        for i in range(0,len(test_data)):
            x = test_data[i,0:4]
            y = test_data[i,4]
            if y == 0:
                y = -1
            if y*(x@w) <= 0:
                error_count += 1
        return error_count/len(test_data)
    def get_voted_error(w_dict,C,m):
        error_count = 0
        test_data = pd.read_csv('./Data/test.csv').to_numpy()
        for i in range(0,len(test_data)):
            x = test_data[i,0:4]
            y = test_data[i,4]
            if y == 0:
                y -= 1
            sum = 0
            for key in range(1,(m+1)):
                if w_dict[key]@x < 0:
                    sum -= C[key]
                else:
                    sum += C[key]
            if y*sum <= 0:
                error_count += 1
        return error_count/len(test_data)
#prints the weight vector/ weight vector table followed by its error on the test data
w_vec = perceptron.standard_perceptron(1)
print('Standard')
print(w_vec)
print(perceptron.get_perceptron_error(w_vec))
print('Voted')
w_vec_dict,C_dict,m = perceptron.voted_perceptron(1)
for key in range(1,(m+1)):
    print("Count: " + str(C_dict[key]) + " Weight Vector: " +str(w_vec_dict[key]))
print(perceptron.get_voted_error(w_vec_dict,C_dict,m))
print('Average')
a_vec = perceptron.average_perceptron(1)
print(a_vec)
print(perceptron.get_perceptron_error(a_vec))
    