import math
import random
#Just copied these classes because the import wasn't working at all and I spent a couple of hours trying to figure it out to no avail. 
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

class Adaboost:
    #returns the vote value and updates the weights
    def err_and_update(data,tree,label):
        error_set = []
        correct_set = []
        error_weight = 0
        for dict in data:
            if DecTree.predict_binary_label(tree,dict) == dict[label]:
                correct_set.append(dict)
            else:
                error_set.append(dict)
                error_weight += dict['w']
        if(error_weight == 0):
            change = 0
        else:
            change = .5*(math.log((1-error_weight)/error_weight))
        err_val = math.exp(change)
        cor_val = math.exp(-change)
        weight_sum = 0
        for err_dat in error_set:
            err_dat['w'] *= err_val
            weight_sum += err_dat['w']
        for corr_dat in correct_set:
            corr_dat['w'] *= cor_val
            weight_sum += corr_dat['w']
        if not weight_sum == 1.0:
            for val in data:
                val['w'] /= weight_sum
        return change
    #finds the number of incorrect predictions of the data using all of the trees and their votes to    
    #make a prediction
    def find_error(data,trees,label,votes):
        error_count = 0
        for dict in data:
            corrPred = 0
            wrongPred = 0
            i = 0
            for tree in trees:
                if DecTree.predict_binary_label(tree,dict) == dict[label]:
                    corrPred += votes[i]
                else:
                    wrongPred += votes[i]
                i += 1
            if corrPred <= wrongPred:
                error_count +=1
        return error_count/len(data)
    # finds the error of the trees and returns a dictionary of the results.
    def tree_error(data,trees,label):
        error_dict = {}
        i = 0
        for tree in trees:
            error_dict[i] = 0
            for dict in data:
                if not DecTree.predict_binary_label(tree,dict) == dict[label]:
                    error_dict[i] += 1
            error_dict[i] /= len(data)
            i += 1
        return error_dict
    #adaboost algorithm that writes the data into Data/AdaBoost_Data.txt
    #writes each stumps test and training errors line by line.
    def adaBoost(train_data,test_data,label):
        attributes = DecTree.make_attributes_array(label)
        size = len(train_data)
        for dict in train_data:
            dict['w'] = 1/size
        forest = []
        votes = []
        f = open("./Data/AdaBoost_Data.txt","w")
        T = 500
        for i in range(0,T):
            tree = Adaboost.build_stump(train_data,attributes,label)
            alpha = Adaboost.err_and_update(train_data,tree,label)
            votes.append(alpha)
            forest.append(tree)
            train_error = Adaboost.find_error(train_data,forest,label,votes)
            test_error = Adaboost.find_error(test_data,forest,label,votes)
            f.write("T: " +str(i) + " train error: "+str(train_error)+" test error: "+ str(test_error)+"\n")
        f.write("Tree errors \n")
        train_dict = Adaboost.tree_error(train_data,forest,label)
        test_dict = Adaboost.tree_error(test_data,forest,label)
        for i in range(1,(T+1)):
            f.write(str(i) +" ")
            if (i-1) in train_dict.keys():
                f.write(str(train_dict[(i-1)]))
            if (i-1) in test_dict.keys():
                f.write(" " + str(test_dict[(i-1)]))
            f.write('\n')
        f.close()
    #returns a dictionary of the values and their weights for the specified variable in the data set
    def make_weighted_variable_dict(data,variable):
        ret_dict = {}
        total = 0
        for dict in data:
            if dict[variable] in ret_dict.keys():
                val = ret_dict[dict[variable]]
                ret_dict[dict[variable]] = val + dict['w']
            else:
                ret_dict[dict[variable]] = dict['w']
            total += dict['w']
        return ret_dict,total
    #returns the variable with the best gain in the data   
    def best_weighted_gain(data,attributes,label):
        label_dict = Adaboost.make_weighted_variable_dict(data,label)
        set_entropy = DecTree.get_entropy(label_dict[1],label_dict[0])
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
                lower_dict = Adaboost.make_weighted_variable_dict(lower_array,label)
                gain -= (lower_dict[1])*DecTree.get_entropy(lower_dict[1],lower_dict[0])
                upper_dict = Adaboost.make_weighted_variable_dict(upper_array,label)
                gain -= (upper_dict[1])*DecTree.get_entropy(upper_dict[1],upper_dict[0])
            else:
                var_dict = Adaboost.make_weighted_variable_dict(data,attribute)
                for key in var_dict[0].keys():
                    subset = DecTree.data_subset(data,attribute,key)
                    subset_dict = Adaboost.make_weighted_variable_dict(subset,label)
                    gain -= (subset_dict[1])*DecTree.get_entropy(subset_dict[1],subset_dict[0])
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute 
    #Builds a stump based on the data and splits on the best gain attribute
    def build_stump(data,attributes,label):
        headNode = Node(None)
        headNode.mcl = DecTree.get_most_common(Adaboost.make_weighted_variable_dict(data,label)[0])
        best_gain = Adaboost.best_weighted_gain(data,attributes,label)
        headNode.data = best_gain
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
                upper_Node.data = headNode.mcl
                upper_Node.mcl = upper_Node.data
            else:
                upper_Node.data = DecTree.get_most_common(Adaboost.make_weighted_variable_dict(higher_subset,label)[0])
                upper_Node.mcl = upper_Node.data
            if len(lower_subset) == 0:
                lower_Node.data = headNode.mcl
                lower_Node.mcl = lower_Node.data
            else:
                lower_Node.data = DecTree.get_most_common(Adaboost.make_weighted_variable_dict(lower_subset,label)[0])
                lower_Node.mcl = lower_Node.data
            headNode.children[threshold] = upper_Node
            headNode.children[-5] = lower_Node
        else:
            split_dict = DecTree.make_variable_dict(data,best_gain)
            for key in split_dict.keys():
                data_subset = DecTree.data_subset(data,best_gain,key)
                newNode = Node(None)
                newNode.data = DecTree.get_most_common(Adaboost.make_weighted_variable_dict(data_subset,label)[0])
                newNode.mcl = newNode.data
                headNode.children[key] = newNode
        return headNode       
class Bagging:
    #generates random data from data with repeats of the same size
    def randData(data):
        random.seed()
        newdata = []
        for val in data:
            newdata.append(data[random.randrange(0,len(data))])
        return newdata
    #generates a random non repeating subset of data that is the size of size
    def subset(data,size):
        random.seed()
        if len(data) < size:
            return data
        subset = []
        chosen = {}
        for i in range(0,size):
            num = random.randrange(0,len(data))
            while num in chosen.keys():
                num = random.randrange(0,len(data))
            subset.append(data[num])
            chosen[num] = 0
        return subset
    #returns 1 if the bagged trees predicts yes, -1 if no
    def bag_prediction(trees,dict):
        yes_count = 0
        for tree in trees:
            if DecTree.predict_binary_label(tree,dict) == 'yes':
                yes_count += 1
        if yes_count/len(trees) >= .5:
            return 1
        else: 
            return -1
    #returns true if the error count is more than half of the trees
    def bagPredictionErr(trees,dict,label):
        error_count = 0
        for tree in trees:
            if not DecTree.predict_binary_label(tree,dict) == dict[label]:
                error_count += 1
        return (error_count/len(trees)) >= .5
    #retuns the training error and testing error of trees
    def BaggingErrors(trees,train_data,test_data,label):
        training_error = 0
        testing_error = 0
        for data in train_data:
            if Bagging.bagPredictionErr(trees,data,label):
                training_error += 1
        for data in test_data:
            if Bagging.bagPredictionErr(trees,data,label):
                testing_error += 1
        return (training_error/len(train_data)),(testing_error/len(test_data))
    #Learns a random tree with a random subset of size size
    def RandTreeLearn(data,attributes,label,node,depth,size):
        node.mcl = DecTree.get_most_common(DecTree.make_variable_dict(data,label))
        if depth == 1 or len(attributes) == 0:  
            node.data = node.mcl  
            return node
        elif DecTree.label_check(data,label):
            node.data = data[0][label]
            node.mcl = node.data
            return node
        else:
            data_subset = Bagging.subset(data,size)
            best_gain = DecTree.find_best_gain_e_binary(data_subset,attributes,label)
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
                    Bagging.RandTreeLearn(higher_subset,attributes,label,upper_Node,depth-1,size)
                    attributes.append(best_gain)
                if len(lower_subset) == 0:
                    lower_Node.data = DecTree.get_most_common(DecTree.make_variable_dict(data,label))
                    lower_Node.mcl = lower_Node.data
                else:
                    attributes.remove(best_gain)
                    Bagging.RandTreeLearn(lower_subset,attributes,label,lower_Node,depth-1,size)
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
                        Bagging.RandTreeLearn(data_subset,attributes,label,newNode,depth-1,size)
                        attributes.append(best_gain)
                    node.children[key] = newNode
            return node
    #gets the bias and variance of the single trees
    def get_bias_var_single(trees,data,label):
        bias = 0
        variance = 0
        for dict in data:
            known_result = dict[label]
            if known_result == 'yes':
                pred = 1
            else:
                pred = -1
            average_tree_pred = 0
            tree_pred_array = []
            for tree in trees:
                tree_pred = DecTree.predict_binary_label(tree,dict)
                if tree_pred == 'yes':
                    average_tree_pred += 1
                    tree_pred_array.append(1)
                else:
                    average_tree_pred -= 1
                    tree_pred_array.append(-1)
            mean = average_tree_pred/len(trees)
            bias += (mean-pred)**2
            var_sum = 0
            for prediction in tree_pred_array:
                var_sum += (prediction - mean)**2
            variance += var_sum/(len(tree_pred_array)-1)
        return bias/len(data),variance/len(data)
    #gets the bias and variance of the bagged trees
    def get_bias_var_bagged(trees_array,data,label):
        bias = 0
        variance = 0
        for dict in data:
            known_result = dict[label]
            if known_result == 'yes':
                pred = 1
            else:
                pred = -1
            average_tree_pred = 0
            tree_pred_array = []
            for trees in trees_array:
                tree_pred = Bagging.bag_prediction(trees,dict)
                average_tree_pred += tree_pred
                tree_pred_array.append(tree_pred)
            mean = average_tree_pred/len(trees_array)
            bias += (mean-pred)**2
            var_sum = 0
            for prediction in tree_pred_array:
                var_sum += (prediction - mean)**2
            variance += var_sum/(len(tree_pred_array)-1)
        return bias/len(data),variance/len(data)
#methods for the hw assignments.
class methods:
    def stump_boost():
        train_data = DecTree.read_file('./bank/train.csv')
        test_data = DecTree.read_file('./bank/test.csv')[0]
        Adaboost.adaBoost(train_data[0],test_data,train_data[1])
    def bagging():
        train_data = DecTree.read_file('./bank/train.csv')
        test_data = DecTree.read_file('./bank/test.csv')
        attribute_array = DecTree.make_attributes_array(train_data[1])
        tree_array = []
        f = open("./Data/Bagging_Data.txt","w")
        for i in range(1,101):
            curr_data = Bagging.randData(train_data[0])
            tree_array.append(DecTree.build_binary_tree(curr_data,attribute_array,train_data[1],Node(None),100,1))
            errors = Bagging.BaggingErrors(tree_array,train_data[0],test_data[0],train_data[1])
            f.write(str(i) + " " + str(errors[0]) + " " + str(errors[1])+ "\n")
        f.close()
    def bagging_RandForest():
        train_data = DecTree.read_file('./bank/train.csv')
        test_data = DecTree.read_file('./bank/test.csv')
        attribute_array = DecTree.make_attributes_array(train_data[1])
        tree_array = []
        f = open("./Data/RandForest_Data2_4_6.txt","w")
        for sub_size in range(2,7,2):
            for i in range(1,101):
                curr_data = Bagging.randData(train_data[0])
                tree_array.append(Bagging.RandTreeLearn(curr_data,attribute_array,train_data[1],Node(None),100,sub_size))
                errors = Bagging.BaggingErrors(tree_array,train_data[0],test_data[0],train_data[1])
                f.write(str(i) + " " + str(errors[0]) + " " + str(errors[1]) + "\n")
        f.close()
    def IntensiveBagging():
        train_data = DecTree.read_file('./bank/train.csv')
        test_data = DecTree.read_file('./bank/test.csv')
        attribute_array = DecTree.make_attributes_array(train_data[1])
        tree_array_array = []
        f = open('./Data/intensive_bagging_bias_var.txt','w')
        for i in range(1,2):
            tree_array = []
            new_data = Bagging.subset(train_data[0],1000)
            for j in range(0,100):
                curr_data = Bagging.randData(new_data)
                tree_array.append(DecTree.build_binary_tree(curr_data,attribute_array,train_data[1],Node(None),100,1))
            tree_array_array.append(tree_array)
        new_tree_array = []
        for TA in tree_array_array:
            new_tree_array.append(TA[0])
        f.write("single tree\n")
        single_tree_bias_var = Bagging.get_bias_var_single(new_tree_array,test_data[0],test_data[1])
        single_tree_general_sqrd_err = single_tree_bias_var[0] + single_tree_bias_var[1]
        f.write(str(single_tree_bias_var[0]) + " " + str(single_tree_bias_var[1]) + " " + str(single_tree_general_sqrd_err) + "\n")
        f.write("bagged trees\n")
        bagged_tree_bias_var = Bagging.get_bias_var_bagged(tree_array_array,test_data[0],test_data[1])
        bagged_tree_general_sqrd_err = bagged_tree_bias_var[0] + bagged_tree_bias_var[1]
        f.write(str(bagged_tree_bias_var[0]) + " " + str(bagged_tree_bias_var[1]) + " " + str(bagged_tree_general_sqrd_err) + "\n")
        f.close()
    def rand_tree_intensive_bagging():
        train_data = DecTree.read_file('./bank/train.csv')
        test_data = DecTree.read_file('./bank/test.csv')
        attribute_array = DecTree.make_attributes_array(train_data[1])
        tree_array_array = []
        f = open('./Data/intensive_rand_tree_bias_var.txt','w')
        for i in range(1,101):
            tree_array = []
            new_data = Bagging.subset(train_data[0],1000)
            for j in range(0,100):
                curr_data = Bagging.randData(new_data)
                tree_array.append(Bagging.RandTreeLearn(curr_data,attribute_array,train_data[1],Node(None),100,4))
            tree_array_array.append(tree_array)
        new_tree_array = []
        for TA in tree_array_array:
            new_tree_array.append(TA[0])
        f.write("single tree\n")
        single_tree_bias_var = Bagging.get_bias_var_single(new_tree_array,test_data[0],test_data[1])
        single_tree_general_sqrd_err = single_tree_bias_var[0] + single_tree_bias_var[1]
        f.write(str(single_tree_bias_var[0]) + " " + str(single_tree_bias_var[1]) + " " + str(single_tree_general_sqrd_err) + "\n")
        f.write("bagged trees\n")
        bagged_tree_bias_var = Bagging.get_bias_var_bagged(tree_array_array,test_data[0],test_data[1])
        bagged_tree_general_sqrd_err = bagged_tree_bias_var[0] + bagged_tree_bias_var[1]
        f.write(str(bagged_tree_bias_var[0]) + " " + str(bagged_tree_bias_var[1]) + " " + str(bagged_tree_general_sqrd_err) + "\n")
        f.close()
methods.IntensiveBagging()
methods.stump_boost()
methods.bagging()
methods.bagging_RandForest()
methods.rand_tree_intensive_bagging()
