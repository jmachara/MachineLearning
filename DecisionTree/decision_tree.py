import math
#Node class
#data is the most common label if it is a leaf node, else it is the variable to split on
#mcl is the most common label of the subset of data for this branch
#children is a dictionary of the children nodes of the node
class Node:
    def __init__(self,data):
        self.data = data
        self.mcl = None
        self.children = {}
class DecTree:
    #Returns if the attribute is numberical for the bank data set
    def is_numerical(attribute):
        num_attributes = [0,5,9,11,12,13,14]
        if attribute in num_attributes:
            return True
        return False
    #reads the file specified in filepath 
    # returns: An array of dictionaries with all of the line data, the dictionary index for the label
    def read_file(filepath):
        data_array = []
        datafile = open(filepath,'r')
        for line in datafile:
            split_values = line.strip().split(',')
            i = 0
            dict = {}
            for values in split_values:
                dict[i] = values
                i+=1
            data_array.append(dict)
        datafile.close()
        return data_array,len(data_array[0])-1
    #returns the median value of the data in the specified varaible 
    def get_median(data,variable):
        values = []
        for dict in data:
            values.append(dict[variable])
        values.sort()
        return values[int(len(values)/2)]
    #Makes an array of the attributes for the data dictionaries
    def make_attributes_array(i):
        return_array = []
        for counter in range(0,i):
            return_array.append(counter)
        return return_array
    #Makes a data subset of the current data with the value of the specifies 'variable' as val
    def data_subset(data,variable,val):
        new_data = []
        for dict in data:
            if dict[variable] == val:
                new_data.append(dict)
        return new_data
    #makes a dictionary of the variables and how many values there are of each
    def make_variable_dict(data,variable):
        ret_dict = {}
        for dict in data:
            if dict[variable] in ret_dict.keys():
                ret_dict[dict[variable]] += 1
            else:
                ret_dict[dict[variable]] = 1
        return ret_dict
    #returns the most common value in the dictionary
    def get_most_common(dict):
        keys = list(dict.keys())
        most_common = keys[0]
        num = dict[keys[0]]
        for i in range(1,len(dict)):
            if dict[keys[i]] > num:
                most_common = keys[i]
                num = dict[keys[i]]
        return most_common
    #checks to see if all the data values have the same label
    def label_check(data,label):
        first_label = data[0][label]
        for dict in data:
            if dict[label] != first_label:
                return False
        return True
    #returns the entropy of the variable that dict has the values of
    #size is the total number of values in dict
    def get_entropy(size,dict):
        entropy = 0
        for key in dict.keys():
            percent = (dict[key]/size)
            entropy -= (percent*math.log2(percent))
        return entropy
    #returns the majority error of the varaible that dict has values of
    def get_majority_error(size,dict):
        if len(dict) == 0:
            return 0
        mcv = DecTree.get_most_common(dict)
        return (size-dict[mcv])/size
    #returns the gini index of the varaible that dict has values of
    def get_gini(size,dict):
        gini = 1
        for key in dict:
            gini -= (dict[key]/size)**2
        return gini
    #finds attribute that provides the best gain to split on in data using entropy
    def find_best_gain_e(data,attributes,label):
        label_dict = DecTree.make_variable_dict(data,label)
        set_entropy = DecTree.get_entropy(len(data),label_dict)
        max_gain = 0
        gain_attribute = attributes[0]
        for attribute in attributes:
            gain = set_entropy
            var_dict = DecTree.make_variable_dict(data,attribute)
            for key in var_dict.keys():
                subset = DecTree.data_subset(data,attribute,key)
                subset_dict = DecTree.make_variable_dict(subset,label)
                gain -= ((var_dict[key]/len(data))*DecTree.get_entropy(len(subset),subset_dict))
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute
    #finds attribute that provides the best gain to split on in data using majority error
    def find_best_gain_me(data,attributes,label):
        label_dict = DecTree.make_variable_dict(data,label)
        set_majority_error = DecTree.get_majority_error(len(data),label_dict)
        max_gain = 0
        gain_attribute = attributes[0]
        for attribute in attributes:
            gain = set_majority_error
            var_dict = DecTree.make_variable_dict(data,attribute)
            for key in var_dict.keys():
                subset = DecTree.data_subset(data,attribute,key)
                subset_dict = DecTree.make_variable_dict(subset,label)
                gain -= ((var_dict[key]/len(data))*DecTree.get_majority_error(len(subset),subset_dict))
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute
    #finds attribute that provides the best gain to split on in data using gini index
    def find_best_gain_g(data,attributes,label):
        label_dict = DecTree.make_variable_dict(data,label)
        set_gini = DecTree.get_gini(len(data),label_dict)
        max_gain = 0
        gain_attribute = attributes[0]
        for attribute in attributes:
            gain = set_gini
            var_dict = DecTree.make_variable_dict(data,attribute)
            for key in var_dict.keys():
                subset = DecTree.data_subset(data,attribute,key)
                subset_dict = DecTree.make_variable_dict(subset,label)
                gain -= ((var_dict[key]/len(data))*DecTree.get_gini(len(subset),subset_dict))
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute
    #finds attribute that provides the best gain to split on in data using entropy
    def find_best_gain_e_binary(data,attributes,label):
        label_dict = DecTree.make_variable_dict(data,label)
        set_entropy = DecTree.get_entropy(len(data),label_dict)
        max_gain = 0
        gain_attribute = attributes[0]
        for attribute in attributes:
            gain = set_entropy
            if DecTree.is_numerical(attribute):
                lower_array = []
                upper_array = []
                threshold = DecTree.get_median(data,attribute)
                for dict in data:
                    if dict[attribute] < threshold:
                        lower_array.append(dict)
                    else:
                        upper_array.append(dict)
                lower_dict = DecTree.make_variable_dict(lower_array,label)
                gain -= (len(lower_array)/len(data))*DecTree.get_entropy(len(lower_array),lower_dict)
                upper_dict = DecTree.make_variable_dict(upper_array,label)
                gain -= (len(upper_array)/len(data))*DecTree.get_entropy(len(upper_array),upper_dict)
            else:
                var_dict = DecTree.make_variable_dict(data,attribute)
                for key in var_dict.keys():
                    subset = DecTree.data_subset(data,attribute,key)
                    subset_dict = DecTree.make_variable_dict(subset,label)
                    gain -= ((var_dict[key]/len(data))*DecTree.get_entropy(len(subset),subset_dict))
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute
    #finds attribute that provides the best gain to split on in data using majority error
    def find_best_gain_me_binary(data,attributes,label):
        label_dict = DecTree.make_variable_dict(data,label)
        set_majority_error = DecTree.get_majority_error(len(data),label_dict)
        max_gain = 0
        gain_attribute = attributes[0]
        for attribute in attributes:
            gain = set_majority_error
            if DecTree.is_numerical(attribute):
                lower_array = []
                upper_array = []
                threshold = DecTree.get_median(data,attribute)
                for dict in data:
                    if dict[attribute] < threshold:
                        lower_array.append(dict)
                    else:
                        upper_array.append(dict)
                lower_dict = DecTree.make_variable_dict(lower_array,label)
                gain -= (len(lower_array)/len(data))*DecTree.get_majority_error(len(lower_array),lower_dict)
                upper_dict = DecTree.make_variable_dict(upper_array,label)
                gain -= (len(upper_array)/len(data))*DecTree.get_majority_error(len(upper_array),upper_dict)
            else:
                var_dict = DecTree.make_variable_dict(data,attribute)
                for key in var_dict.keys():
                    subset = DecTree.data_subset(data,attribute,key)
                    subset_dict = DecTree.make_variable_dict(subset,label)
                    gain -= ((var_dict[key]/len(data))*DecTree.get_majority_error(len(subset),subset_dict))
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute
    #finds attribute that provides the best gain to split on in data using gini index
    def find_best_gain_g_binary(data,attributes,label):
        label_dict = DecTree.make_variable_dict(data,label)
        set_gini = DecTree.get_gini(len(data),label_dict)
        max_gain = 0
        gain_attribute = attributes[0]
        for attribute in attributes:
            gain = set_gini
            if DecTree.is_numerical(attribute):
                lower_array = []
                upper_array = []
                threshold = DecTree.get_median(data,attribute)
                for dict in data:
                    if dict[attribute] < threshold:
                        lower_array.append(dict)
                    else:
                        upper_array.append(dict)
                lower_dict = DecTree.make_variable_dict(lower_array,label)
                gain -= (len(lower_array)/len(data))*DecTree.get_gini(len(lower_array),lower_dict)
                upper_dict = DecTree.make_variable_dict(upper_array,label)
                gain -= (len(upper_array)/len(data))*DecTree.get_gini(len(upper_array),upper_dict)
            else:
                var_dict = DecTree.make_variable_dict(data,attribute)
                for key in var_dict.keys():
                    subset = DecTree.data_subset(data,attribute,key)
                    subset_dict = DecTree.make_variable_dict(subset,label)
                    gain -= ((var_dict[key]/len(data))*DecTree.get_gini(len(subset),subset_dict))
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute
    #builds the tree using the data provided. returns the root node
    def build_tree(data,attributes,label,node,depth,split):
        node.mcl = DecTree.get_most_common(DecTree.make_variable_dict(data,label))
        if depth == 1 or len(attributes) == 0:  
            node.data = node.mcl  
            return node
        elif DecTree.label_check(data,label):
            node.data = data[0][label]
            return node
        else:
            if split == 1:
                best_gain = DecTree.find_best_gain_e(data,attributes,label)
            elif split == 2:
                best_gain = DecTree.find_best_gain_me(data,attributes,label)
            else:
                best_gain = DecTree.find_best_gain_g(data,attributes,label)
            node.data = best_gain
            split_dict = DecTree.make_variable_dict(data,best_gain)
            for key in split_dict.keys():
                data_subset = DecTree.data_subset(data,best_gain,key)
                newNode = Node(None)
                if len(data_subset) == 0:
                    newNode.data = DecTree.get_most_common(DecTree.make_variable_dict(data,label))
                else:
                    attributes.remove(best_gain)
                    DecTree.build_tree(data_subset,attributes,label,newNode,depth-1,split)
                    attributes.append(best_gain)
                node.children[key] = newNode
            return node
    #builds the tree using the data provided. returns the root node
    def build_binary_tree(data,attributes,label,node,depth,split):
        node.mcl = DecTree.get_most_common(DecTree.make_variable_dict(data,label))
        if depth == 1 or len(attributes) == 0:  
            node.data = node.mcl  
            return node
        elif DecTree.label_check(data,label):
            node.data = data[0][label]
            node.mcl = node.data
            return node
        else:
            if split == 1:
                best_gain = DecTree.find_best_gain_e_binary(data,attributes,label)
            elif split == 2:
                best_gain = DecTree.find_best_gain_me_binary(data,attributes,label)
            else:
                best_gain = DecTree.find_best_gain_g_binary(data,attributes,label)
            node.data = best_gain
            if DecTree.is_numerical(best_gain):
                threshold = DecTree.get_median(data,best_gain)
                higher_subset = []
                lower_subset = []
                for dict in data:
                    if dict[best_gain] < threshold:
                        lower_subset.append(dict)
                    else:
                        higher_subset.append(dict)
                upper_Node = Node(None)
                lower_Node = Node(None)
                if len(higher_subset) == 0:
                    upper_Node.data = DecTree.get_most_common(DecTree.make_variable_dict(data,label))
                    upper_Node.mcl = upper_Node.data
                else:
                    attributes.remove(best_gain)
                    DecTree.build_binary_tree(higher_subset,attributes,label,upper_Node,depth-1,split)
                    attributes.append(best_gain)
                if len(lower_subset) == 0:
                    lower_Node.data = DecTree.get_most_common(DecTree.make_variable_dict(data,label))
                    lower_Node.mcl = lower_Node.data
                else:
                    attributes.remove(best_gain)
                    DecTree.build_binary_tree(lower_subset,attributes,label,lower_Node,depth-1,split)
                    attributes.append(best_gain)
                node.children[threshold] = upper_Node
                node.children[-5] = lower_Node
            else:
                split_dict = DecTree.make_variable_dict(data,best_gain)
                for key in split_dict.keys():
                    data_subset = DecTree.data_subset(data,best_gain,key)
                    newNode = Node(None)
                    if len(data_subset) == 0:
                        newNode.data = DecTree.get_most_common(DecTree.make_variable_dict(data,label))
                        newNode.mcl = newNode.data
                    else:
                        attributes.remove(best_gain)
                        DecTree.build_binary_tree(data_subset,attributes,label,newNode,depth-1,split)
                        attributes.append(best_gain)
                    node.children[key] = newNode
            return node
    #returns an array of trees and an array of labels with every combination from the splits and depths on the file
    def build_tree_array(splits,depths,file):
        tree_array = []
        label = []
        for split in splits:
            for depth in depths:
                data_label = DecTree.read_file(file)
                attribute_array = DecTree.make_attributes_array(data_label[1])
                tree_array.append(DecTree.build_tree(data_label[0],attribute_array,data_label[1],Node(None),depth,split))
                label.append(data_label[1])
        return tree_array,label
    
        #returns a cleaned version of data, where 'unknown' is replaced with the most common value
    #used to removed 'unknown' from the data set
    def clean_data(data,attributes):
        for key in attributes:
            varDict = DecTree.make_variable_dict(data,key)
            varDict['unknown'] = 0
            mcv = DecTree.get_most_common(varDict)
            for dict in data:
                if dict[key] == 'unknown':
                    dict[key] = mcv
        return data
    #builds an array of trees and labels with the combination from the splits and depths in the file. b specifies if the data should be cleaned
    def build_binary_tree_array(splits,depths,file,b):
        tree_array = []
        label = []
        for split in splits:
            for depth in depths:
                data_label = DecTree.read_file(file)
                attribute_array = DecTree.make_attributes_array(data_label[1])
                if b:
                    clean_data = DecTree.clean_data(data_label[0],attribute_array)
                    tree_array.append(DecTree.build_binary_tree(clean_data,attribute_array,data_label[1],Node(None),depth,split))
                else:
                    tree_array.append(DecTree.build_binary_tree(data_label[0],attribute_array,data_label[1],Node(None),depth,split))
                label.append(data_label[1])
        return tree_array,label
    # returns if the value of the predicted label
    def predict_binary_label(tree,dict):
        current_node = tree
        while len(current_node.children) > 0:
            var = current_node.data
            branch = dict[var]
            if DecTree.is_numerical(var):
                for key in current_node.children.keys():
                    if key != -5:
                        if branch < key:
                            current_node = current_node.children[-5]
                        else:
                            current_node = current_node.children[key]
            elif branch in current_node.children.keys():
                current_node = current_node.children[branch]
            else:
                return current_node.mcl
        return current_node.data
    # returns if the value of the label is the same as the predicted value from the tree.
    def predict_label(tree,dict):
        current_node = tree
        while len(current_node.children) > 0:
            var = current_node.data
            branch = dict[var]
            if branch in current_node.children.keys():
                current_node = current_node.children[branch]
            else:
                return current_node.mcl
        return current_node.data 
    #returns the number of lines that don't get the correct prediction from the tree
    #var has an array of trees as its 0 index, and the array of labels in its second index
    def get_prediction_errors(var,testing_data,binary):
        i = 0
        return_array = []
        for tree in var[0]:
            error_counter = 0
            label = var[1][i]
            i +=1
            data_length = len(testing_data)
            if binary:
                for dict in testing_data:
                    if not DecTree.predict_binary_label(tree,dict) == dict[label]:
                        error_counter+=1
                return_array.append(error_counter/data_length)
            else:
                for dict in testing_data:
                    if not DecTree.predict_label(tree,dict) == dict[label]:
                        error_counter+=1
                return_array.append(error_counter/data_length)
        return return_array
#This class has all of the main running methods for the problems.
class methods:
    def get_car_tree_errors():
        test_splits = [1,2,3]
        split_names = ['information_gain','majority_error', 'gini_index']
        test_depths = [1,2,3,4,5,6]
        training_file = './DecisionTree/car/train.csv'
        test_files = ['./DecisionTree/car/train.csv','./DecisionTree/car/test.csv']

        tree_array__label = DecTree.build_tree_array(test_splits,test_depths,training_file)
        testing_data = []
        i = 0
        for file in test_files:
            testing_data.append(DecTree.read_file(file)[0])
        error_array = []
        for data in testing_data:
            error_array.append(DecTree.get_prediction_errors(tree_array__label,data,False))
        f = open("./DecisionTree/Data/Car_Data.txt","w")
        i = 0
        for file in test_files:
            j = 0
            for split in split_names:
                for depth in test_depths:
                    f.write("file: "+file +"    split: "+split+"    depth: "+str(depth)+"   error %: "+str(error_array[i][j]) + "\n")
                    j+=1
                f.write("\n")
            i+=1
        f.close()
    def get_bank_tree_errors(b):
        #test_splits = [1,2,3]
        test_splits = [1]
        split_names = ['information_gain','majority_error', 'gini_index']
        test_depths = [1,2]
        #test_depths = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        training_file = './DecisionTree/bank/train.csv'
        test_files = ['./DecisionTree/bank/train.csv','./DecisionTree/bank/test.csv']

        tree_array__label = DecTree.build_binary_tree_array(test_splits,test_depths,training_file,b)
        testing_data = []
        i = 0
        for file in test_files:
            td = DecTree.read_file(file)
            if b:
                testing_data.append(DecTree.clean_data(td[0],DecTree.make_attributes_array(td[1])))
            else:
                testing_data.append(td[0])
        error_array = []
        for data in testing_data:
            error_array.append(DecTree.get_prediction_errors(tree_array__label,data,True))
        if b:
            f = open("./Data/Bank_Data_b.txt","w")
        else:
            f = open("./DecisionTree/Data/Bank_Data.txt","w")
        i = 0
        for file in test_files:
            j = 0
            for split in split_names:
                for depth in test_depths:
                    f.write("file: "+file +"    split: "+split+"    depth: "+str(depth)+"   error %: "+str(error_array[i][j]) + "\n")
                    j+=1
                f.write("\n\n")
            i+=1
        f.close()
methods.get_car_tree_errors()
methods.get_bank_tree_errors(False)
methods.get_bank_tree_errors(True)
