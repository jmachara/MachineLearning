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
