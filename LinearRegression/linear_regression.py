import numpy as np
import math
import random as rnd
#linear regression for homework 2

class linear_regression:
    #Reads the training file
    def get_xmat_ymat(filepath):
        f = open(filepath,'r')
        t_x_mat = np.empty([53,7])
        y_mat = np.empty([53,1])
        index = 0
        for line in f:
            split_values = line.strip().split(',')
            vector = []
            for i in range(0,7):
                vector.append(split_values[i])
            t_x_mat[index] = vector
            y_mat[index] = split_values[7]
            index +=1   
        f.close()
        return t_x_mat.T,y_mat
    #reads the test file
    def get_test_xmat_ymat(filepath):
        f = open(filepath,'r')
        t_x_mat = np.empty([50,7])
        y_mat = np.empty([50,1])
        index = 0
        for line in f:
            split_values = line.strip().split(',')
            vector = []
            for i in range(0,7):
                vector.append(split_values[i])
            t_x_mat[index] = vector
            y_mat[index] = split_values[7]
            index +=1   
        f.close()
        return t_x_mat.T,y_mat
    #calculates the optimal weight vector for the concrete data
    def analytical_weight():
        xmat_ymat = linear_regression.get_xmat_ymat('./concrete/train.csv')
        inv_x = np.linalg.inv(xmat_ymat[0]@xmat_ymat[0].T)
        x_mult = inv_x@xmat_ymat[0]
        w = np.dot(x_mult,xmat_ymat[1])
        f = open('./Data/analytical_weight','w')
        f.write(str(w))
        f.close()
    #computes the gradient for batch gradient
    def compute_grad(w,x_arr,examples,y):
        guesses = np.sum(np.multiply(w,examples),axis=1)
        errors = np.subtract(y.T,guesses)
        errors = np.multiply(errors,x_arr)
        return -np.sum(errors)
    #returns the cost of w on the examples
    def get_cost(w,examples,y):
        guesses = np.sum(np.multiply(w,examples),axis=1)
        errors = np.subtract(y.T,guesses)
        return np.sum(errors**2)/2   
    #returns the new w vector
    def update_w_sto(w,example,y,r):
        error = np.subtract(y,np.sum(np.multiply(w,example)))
        change = np.multiply(error,example)
        new_w = np.add(w,r*change)
        return new_w
    #Writes the cost of each loop on the train data, followed by the cost on the test data and w vector in 
    #Data/batch_grad_cost.txt
    def batch_grad_des(r):
        w_vec = np.zeros(7)
        xmat_ymat = linear_regression.get_xmat_ymat('./concrete/train.csv')
        test_xmat_ymat = linear_regression.get_test_xmat_ymat('./concrete/test.csv')
        vec_diff = 1
        f = open('./Data/batch_grad_cost','w')
        while vec_diff > .000001:
            f.write(str(linear_regression.get_cost(w_vec,xmat_ymat[0].T,xmat_ymat[1])) + "\n")
            grad_vec = np.empty(7)
            for i in range(0,7):
                grad_vec[i] = linear_regression.compute_grad(w_vec,xmat_ymat[0][i],xmat_ymat[0].T,xmat_ymat[1])
            new_w_vec = np.subtract(w_vec,r*grad_vec)
            vec_diff = math.sqrt(np.sum(np.subtract(new_w_vec.T,w_vec)**2))
            w_vec= new_w_vec
        f.write('w vector \n')
        f.write(str(w_vec))
        f.write('cost on test data\n'+ str(linear_regression.get_cost(w_vec,test_xmat_ymat[0].T,test_xmat_ymat[1])))
        f.close()
    #Writes the cost of each loop on the train data, followed by the cost on the test data and w vector in 
    #Data/stoch_grad_cost.txt
    def stochastic_grad_des(r):
        rnd.seed()
        w_vec = np.zeros(7)
        xmat_ymat = linear_regression.get_xmat_ymat('./concrete/train.csv')
        test_xmat_ymat = linear_regression.get_xmat_ymat('./concrete/test.csv')
        prev_cost = linear_regression.get_cost(w_vec,xmat_ymat[0].T,xmat_ymat[1])
        cost_diff = 1
        f = open('./Data/stoch_grad_cost','w')
        while cost_diff > .0000001:
            f.write(str(linear_regression.get_cost(w_vec,xmat_ymat[0].T,xmat_ymat[1])) + "\n")
            i = rnd.randrange(0,len(xmat_ymat[0].T))
            new_w_vec = linear_regression.update_w_sto(w_vec,xmat_ymat[0].T[i],xmat_ymat[1][i],r)
            cost = linear_regression.get_cost(new_w_vec,xmat_ymat[0].T,xmat_ymat[1])
            cost_diff = abs(prev_cost-cost)
            prev_cost = cost
            w_vec = new_w_vec
        f.write('test cost \n')
        f.write(str(linear_regression.get_cost(w_vec,test_xmat_ymat[0].T,test_xmat_ymat[1])) + "\n")
        f.write('w vector \n')
        f.write(str(w_vec))
        f.close()
linear_regression.batch_grad_des(.01)
linear_regression.stochastic_grad_des(.01)
linear_regression.analytical_weight()
