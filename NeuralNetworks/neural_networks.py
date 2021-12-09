import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt 
class NeuralNetwork:
    #Initializes the network, has np arrays to hold weights and z values. 
    def __init__(self,inputnum,width):
        self.weights_l1 = np.zeros((inputnum+1,width))
        self.weights_l2 = np.zeros((width,width))
        self.weights_l3 = np.zeros(width)
        self.input_vals = np.zeros(inputnum+1)
        self.z_vals_l1 = np.zeros(width)
        self.z_vals_l1[0] = 1
        self.z_vals_l2 = np.zeros(width)
        self.z_vals_l2[0] = 1
        self.y = 0
        self.width = width
        self.inputs = inputnum+1
    #Computes the values of the z nodes and the final y with the given input vals in the neural network
    def compute_z_y_values(self):
        for i in range(1,self.width):
            self.z_vals_l1[i] = 1/(1+math.exp(-np.dot(self.weights_l1[:,i],self.input_vals)))
        for i in range(1,self.width):
            self.z_vals_l2[i] = 1/(1+math.exp(-np.dot(self.weights_l2[:,i],self.z_vals_l1)))
        self.y = np.dot(self.weights_l3,self.z_vals_l2)
    #randomizes the edge weights
    def random_edges(self):
        self.weights_l1 = np.random.rand(self.inputs,self.width)
        self.weights_l2 = np.random.rand(self.width,self.width)
        self.weights_l3 = np.random.rand(self.width)
class NN:
    #calculates the mean loss of the network on the data
    def calc_loss(network,data):
        sum = 0
        for d in data:
            y_star = d[-1]
            if y_star == 0:
                y_star = -1
            network.input_vals = np.concatenate(([1],d[:-1]))
            network.compute_z_y_values()
            sum += .5*(network.y-y_star)**2
        return sum/len(data)
    #trains the network on the training data with the given gamma and d values. 
    def train(network,gamma,d,epoch):
        training_data = pd.read_csv('./Data/train.csv')
        loss = np.empty(epoch)
        for t in range(epoch):
            gamma_t = gamma/(1+(gamma/d)*t)
            rand_data = training_data.sample(frac=1).to_numpy()
            for data in rand_data:
                y_star = data[-1]
                if y_star == 0:
                    y_star = -1
                network.input_vals = np.concatenate(([1],data[:-1]))
                network.compute_z_y_values()
                grad1,grad2,grad3 = NN.backprop(network,y_star)
                network.weights_l3 -= gamma_t*grad3
                network.weights_l2 -= gamma_t*grad2
                network.weights_l1 -= gamma_t*grad1
            loss_val = NN.calc_loss(network,training_data.to_numpy())
            loss[t] = loss_val
        plt.plot(np.linspace(1,epoch,epoch),loss)
        plt.show(block=True)
    #returns the test and training error of the network
    def calc_error(network):
        training_data = pd.read_csv('./Data/train.csv').to_numpy()
        test_data = pd.read_csv('./Data/test.csv').to_numpy()
        training_error = 0
        test_error = 0
        for d in training_data:
            network.input_vals = np.concatenate(([1],d[:-1]))
            network.compute_z_y_values()
            if network.y > 0:
                if d[-1] == 0:
                    training_error+=1
            else:
                if d[-1] == 1:
                    training_error+=1
        for d in test_data:
            network.input_vals = np.concatenate(([1],d[:-1]))
            network.compute_z_y_values()
            if network.y > 0:
                if d[-1] == 0:
                    test_error+=1
            else:
                if d[-1] == 1:
                    test_error+=1
        return training_error/len(training_data),test_error/len(test_data)
    #builds the network to that from homwork 5 for testing purposes
    def build_hw_nn():
        network = NeuralNetwork(2,3)
        network.weights_l1 = np.array(([0,-1,1],[0,-2,2],[0,-3,3]))
        network.weights_l2 = np.array(([0,-1,1],[0,-2,2],[0,-3,3]))
        network.weights_l3 = np.array([-1,2,-1.5])
        network.input_vals = np.array([1,1,1])
        network.compute_z_y_values()
        return network
    #returns the gradient for each of the 3 layers using back propagation
    def backprop(network,y_star):
        layer3_grad = NN.comp_grad_l3(network,y_star)
        layer2_grad = NN.comp_grad_l2(network,y_star)
        layer1_grad = NN.comp_grad_l1(network,y_star)
        return layer1_grad,layer2_grad,layer3_grad
    #computes the level 1 gradient
    def comp_grad_l1(network,y_star):
        gradient = np.zeros(network.weights_l1.shape)
        dl_dy = network.y-y_star
        for i in range(0,network.inputs):
            for j in range(1,network.width):
                for k in range(1,network.width):
                    z1 = network.z_vals_l1[j]
                    z2 = network.z_vals_l2[k]
                    gradient[i,j] += dl_dy*network.weights_l3[k]*(z2*(1-z2))*network.weights_l2[j,k]*(z1*(1-z1))*network.input_vals[i]
        return gradient
    #computes the level 2 gradient
    def comp_grad_l2(network,y_star):
        gradient = np.zeros(network.weights_l2.shape)
        dl_dy = network.y-y_star
        for i in range(0,network.width):
            for j in range(1,network.width):
                z = network.z_vals_l2[j]
                gradient[i,j] = dl_dy*network.weights_l3[j]*(z*(1-z))*network.z_vals_l1[i]
        return gradient
    #computes the level 3 gradient
    def comp_grad_l3(network,y_star):
        gradient = np.zeros(network.weights_l3.shape)
        dl_dy = network.y-y_star
        i = 0
        for val in network.z_vals_l2:
            gradient[i] = val*dl_dy
            i+=1
        return gradient

#part b
print("part b")
for i in [5,10,25]:
    network = NeuralNetwork(4,i)
    network.random_edges()
    NN.train(network,.05,.1,100)
    train_err,test_err = NN.calc_error(network)
    print("width = " + str(i) + " train_error = " + str(train_err) + " test_error = " + str(test_err))   
for i in [50, 100]:
    network = NeuralNetwork(4,i)
    network.random_edges()
    NN.train(network,.005,.1,100)
    train_err,test_err = NN.calc_error(network)
    print("width = " + str(i) + " train_error = " + str(train_err) + " test_error = " + str(test_err)) 
#part c
print("part c")
for i in [5, 10, 25]:
    network = NeuralNetwork(4,i)
    NN.train(network,.05,.1,100)
    train_err,test_err = NN.calc_error(network)
    print("width = " + str(i) + " train_error = " + str(train_err) + " test_error = " + str(test_err))
for i in [50, 100]:
    network = NeuralNetwork(4,i)
    NN.train(network,.0005,.1,100)
    train_err,test_err = NN.calc_error(network)
    print("width = " + str(i) + " train_error = " + str(train_err) + " test_error = " + str(test_err))    

