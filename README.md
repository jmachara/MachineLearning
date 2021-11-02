# MachineLearning Fall 2021
This is a machine learning library developed by Jack Machara for
CS 5350 in University of Utah

Using DecisionTree:
  Learning a Tree:
    The steps to learning a tree with my code are as follows:
    
  1. Use Algorithms.read_file() with the filename as its parameter to get a list of the data in dictionaries as the first index of the return value, and the dictionary index
        for the label of the data as the second value. 
        
        
  2. Call the Algorithms.make_attributes_array() with the second index of the read_file() return, which is the dicitonary key for the label. This returns an array of the 
          attribute key values in the dictionaries. 
          
          
  3. Call the Algorithms.build_tree() with 6 parameters to learn the tree: the list of data dictionaries from read_file(), the attributes array, label index/second return of read_file(), Node(none), the depth you want to make the tree, and an integer based on what split to use: 1 for information gain, 2 for majority error, any other int for gini index. 
      
      
  4. The build_tree() method will return the empty node passed in as the head node of the learned tree which then can be used as needed. 
     
     
  5. Each node in the tree has 3 variables: data which is the dictionary key of the data to split on, mcl which is the most common label of the outcomes at 
         this point in the tree, and children which is a dictionary of the branches. 
         
To learn the tree with binary data, use the method build_binary_tree() instead. Build_tree_array can be used to create many trees with varying data. The parameters are an array of splits, array of depths, and the array of files. 

Homework 2:

  You can run the code with the run.sh file provided, which moves the current directory into the file's folder and runs it to produce the text files. If you want to use the code without the run.sh file, follow the steps below
  
 Adaboost:
    To run the Adaboost code, call the stump_boost method in the methods class. This calls the adaboost algorithm on the bank data
    
    
Bagged Trees:
    To run the bagged trees code, call the bagging method in the methods class. This builds the tree groups and tests them. 


Random Forest:
        To run the random forest code, call the bagging_RandForest method in the methods class. This builds the rand tree groups and tests them. 


Intensive tests:
        The intensive bagged trees test is ran using the IntensiveBagging method in the methods class. For the random tree intensive method, run rand_tree_intensive_bagging.
        This generates specification of the intensive testing from homework 2, but instead of 500, I used 100 due to time limitations. 


Batch Gradient:
        This can be called using the batch_grad_des method that has a parameter for the step. .01 was the chosen value for this parameter. 


Stochastic Gradient:
        This can be called using the stochastic_grad_des method that has a parameter for the step. .01 was the chosen value for this parameter.


Analytical:
        To calculate the best weight vector analytically, use the analytical_weight function. 
