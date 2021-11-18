
import numpy as np
from numpy.core.function_base import linspace
import pandas as pd
from pandas.io.formats.format import return_docstring
from scipy import optimize
from matplotlib import pyplot as plt 

class SVM:
    #returns the error with a weight vec and the data
    def get_error(w,data):
        error_count = 0
        for i in range(0,len(data)):
            x = np.copy(data[i,:])
            y = data[i,4]
            if y == 0:
                y = -1
            x[-1] = 1
            if y*(x@w) <= 0:
                error_count += 1
        return error_count/len(data)
    #finds which vectors are support vectors
    def get_supp_vecs(alpha_vals):
        supp_dict = {}
        for i in range(len(alpha_vals)):
            if alpha_vals[i] == 0:
                supp_dict[i] = 1
        return supp_dict
    #compares support vec dictionaries, returns number of the same key
    def comp_supp_vecs(vec,vec2):
        same_count = 0
        for key in vec.keys():
            if key in vec2.keys():
                same_count+=1
        return same_count
    #returns the error for kernal prediction
    def get_kernel_error(pred_vec,bias,y):
        error_count = 0
        for i in range(0,len(y)):
            if y[i]*(pred_vec[i]+bias) <=0:
                error_count += 1
        return error_count/len(y)
    #Returns the weight vector and bias of a stochastic svm with the inputted parameters
    #C: C value
    #l_0: learning rate
    #a: a value for l_t
    #partA: true if it is part A for homework 4 question 2, used to cover a and b in one function.
    def stochastic_svm(C,l_0,a,partA):
        data = pd.read_csv('./Data/train.csv')
        w = np.zeros(5)
        b = 0
        N = len(data)
        #costs = np.empty(100)
        for t in range(0,100):
            if partA:
                l_t = l_0/(1+((l_0*t)/a))
            else:
                l_t = l_0/(1+t)
            rand_data = data.sample(frac=1).to_numpy()
            for i in range(0,N):
                x = np.copy(rand_data[i,:])
                y = rand_data[i,-1]
                if y == 0:
                    y = -1
                x[-1] = 1
                if (x@w)*y <= 1:
                    w_0 = w.copy()
                    w_0[-1] = 0
                    w = w-(l_t*w_0)+ (l_t*C*N*y*x)
                else:
                        w = (1-l_t)*w
        #checking for convergence
            #costs[t] = SVM.get_cost(data.to_numpy(),w,C)
        #plt.plot(np.linspace(1,100,100),costs)
        #plt.show()
        return w
    #faster solution used was found at https://stackoverflow.com/questions/61022308/converting-double-for-loop-to-numpy-linear-algebra-for-dual-form-svm
    def svm_dual(alpha,training_data):
        x = training_data[:,:-1]
        y = np.where(training_data[:,-1] == 0,-1,1)
        first_sum = np.einsum("i,j,i,j,ix,jx->",y,y,alpha,alpha,x,x,optimize="optimal")/2
        return first_sum-np.sum(alpha)
        # faster solution for kernal at https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python
    
    def svm_dual_kernal(alpha,training_data,gamma):
        x = training_data[:,:-1]
        y = np.where(training_data[:,-1] == 0,-1,1)
        norm= np.einsum('ij,ij->i',x,x)
        kernal = np.exp(-(norm[:,None] + norm[None,:] - 2 * np.dot(x, x.T))/gamma)
        first_sum = np.einsum("i,j,i,j,ij->",y,y,alpha,alpha,kernal,optimize="optimal")/2
        return first_sum-np.sum(alpha)
    #returns the cost of the given weight vector, bias and C from the given data. 
    def get_cost(data,w,C):
        y_vec = np.where(data[:,-1] == 0,-1,1)
        x_vec = np.copy(data)
        x_vec[:,-1] = np.ones(np.shape(y_vec))
        w_0 = w[:-1]
        vec_product = np.dot(w_0,w_0)/2
        summation = C*np.sum(np.maximum(0,1-y_vec * np.dot(x_vec,w)))
        return vec_product+summation
    #gets the weight vector and bias from the alpha vector
    def get_w_b(alpha_vec,x_vec,y_vec):
        a_y = alpha_vec*y_vec
        weight_vec = np.sum(a_y[:,None]*x_vec,axis=0)
        bias = np.sum(y_vec-np.dot(x_vec,weight_vec))/len(y_vec)
        return weight_vec,bias
    #gets predictions from the dual svm kernal method
    def get_k_pred(alpha_vec,training_x,training_y,test_x, gamma):
        return_arr = np.empty(len(test_x[:,0]))
        for i in range(len(test_x)):
            x_vec = np.repeat(test_x[i][np.newaxis,:],training_x.shape[0],axis=0)
            kernal = np.exp(-np.square(np.linalg.norm(training_x  - x_vec,axis=1))/gamma)
            return_arr[i] = np.sum(alpha_vec*training_y*kernal)
        return return_arr

class Methods:
    #prints the C value along with the test and training error to the console.
    def dual_svm(pb):
        test_data = pd.read_csv('./Data/test.csv').to_numpy()  
        training_data = pd.read_csv('./Data/train.csv').to_numpy() 
        x_data = training_data[:,:-1]
        y_data = np.where(training_data[:,-1] == 0,-1,1)
        y_test = np.where(test_data[:,-1] == 0,-1,1)
        C_vec = [100/873,500/873,700/873]
        #for part c
        #C_vec = [500/873]
        gamma = [.1,.5,1,5,100]
        sup_vec_list = []
        if pb:
            print("Dual SVM Kernal")
            loop = 1
            for C_val in C_vec:
                for g in gamma:
                    cons = ({'type': 'eq', 'fun': lambda x,y: np.dot(x,y),'args': [y_data]})
                    bs = optimize.Bounds(0,C_val)
                    result = optimize.minimize(SVM.svm_dual_kernal,np.zeros(871),(training_data,g),method='SLSQP',bounds=bs,constraints=cons)
                    pred_vec_train = SVM.get_k_pred(result.x,x_data,y_data,x_data,g)
                    pred_vec_test = SVM.get_k_pred(result.x,x_data,y_data,test_data[:,:-1],g)
                    bias_train = np.mean(y_data-pred_vec_train)
                    bias_test = np.mean(y_test-pred_vec_test)                    
                    test_err = SVM.get_kernel_error(pred_vec_test,bias_test,y_test)
                    train_err = SVM.get_kernel_error(pred_vec_train,bias_train,y_data)
                    supp_vec_dict = SVM.get_supp_vecs(result.x)
                    if loop == 2:
                        sup_vec_list.append(supp_vec_dict) #indent when uncommenting
                    print("C = " + str(C_val) + " gamma = " + str(g)  + " test err:" + str(test_err) + " train err: " + str(train_err) + " support vec count: " + str(len(supp_vec_dict.keys())))
                    print()
                loop += 1
            print("support vec comparison")
            for i in range(0,len(sup_vec_list)-1):
                print(str(gamma[i]) + " " + str(gamma[i+1]))
                print(SVM.comp_supp_vecs(sup_vec_list[i],sup_vec_list[i+1]))
            return
        print("Dual SVM")
        for C_val in C_vec:
            cons = ({'type': 'eq', 'fun': lambda x,y: np.dot(x,y),'args': [y_data]})
            bs = optimize.Bounds(0,C_val)
            result = optimize.minimize(SVM.svm_dual,np.zeros(871),training_data,method='SLSQP',bounds=bs,constraints=cons)
            w,b = SVM.get_w_b(result.x,x_data,y_data)
            test_err = SVM.get_error(np.append(w,b),test_data)
            train_err = SVM.get_error(np.append(w,b),training_data)
            print("C = " + str(C_val) + " w_vec = " + str(w) + " bias = " + str(b) + " test err:" + str(test_err) + " train err: " + str(train_err))
    def stoc_svm_():
        test_data = pd.read_csv('./Data/test.csv').to_numpy()
        training_data = pd.read_csv('./Data/train.csv').to_numpy()        
        C_vec = [100/873,500/873,700/873]
        l_val = [.1]
        a_val = [.1]
        val = True
        print("Part A:")
        for i in range(0,2):
            for a in a_val:
                for l in l_val:
                    for C_val in C_vec:
                        w = SVM.stochastic_svm(C_val,l,a,val)
                        test_err = SVM.get_error(w,test_data)
                        train_err = SVM.get_error(w,training_data)
                        print("weight vector: " + str(w))
                        print("C = " + str(C_val) +" a = " + str(a) +" l = " + str(l) + " train error = " + str(train_err) + " testing error = "  + str(test_err))
            if val:
                print("Part B:")
            val = False
print(2)
Methods.stoc_svm_()
print(3)
print('a')
Methods.dual_svm(False)
print('b')
Methods.dual_svm(True)

    

