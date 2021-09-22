import math
class Node:
    def __init__(self,data):
        self.data = data
        self.children = {}

class Algorithms:
    
    
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
    
    
    def make_attributes_array(i):
        return_array = []
        for counter in range(0,i):
            return_array.append(counter)
        return return_array
    
    
    def data_subset(data,variable,val):
        new_data = []
        for dict in data:
            if dict[variable] == val:
                new_data.append(dict)
        return new_data
    
    
    def make_variable_dict(data,variable):
        ret_dict = {}
        for dict in data:
            if dict[variable] in ret_dict.keys():
                ret_dict[dict[variable]] += 1
            else:
                ret_dict[dict[variable]] = 1
        return ret_dict
    
    
    def get_most_common(dict):
        keys = list(dict.keys())
        most_common = keys[0]
        num = dict[keys[0]]
        for i in range(1,len(dict)):
            if dict[keys[i]] > num:
                most_common = keys[i]
                num = dict[keys[i]]
        return most_common
    
    
    def label_check(data,label):
        first_label = data[0][label]
        for dict in data:
            if dict[label] != first_label:
                return False
        return True
    
    
    def get_entropy(size,dict):
        entropy = 0
        for key in dict.keys():
            percent = (dict[key]/size)
            entropy -= (percent*math.log2(percent))
        return entropy
    
    
    def get_majority_error(size,dict):
        majority_error = 0
        mcv = Algorithms.get_most_common(dict)
        return (size-dict[mcv])/size
    
    
    def get_gini(size,dict):
        gini = 1
        for key in dict:
            gini -= (dict[key]/size)**2
        return gini
    
    
    def find_best_gain_e(data,attributes,label):
        label_dict = Algorithms.make_variable_dict(data,label)
        set_entropy = Algorithms.get_entropy(len(data),label_dict)
        max_gain = 0
        gain_attribute = attributes[0]
        for attribute in attributes:
            gain = set_entropy
            var_dict = Algorithms.make_variable_dict(data,attribute)
            for key in var_dict.keys():
                subset = Algorithms.data_subset(data,attribute,key)
                subset_dict = Algorithms.make_variable_dict(subset,label)
                gain -= ((var_dict[key]/len(data))*Algorithms.get_entropy(len(subset),subset_dict))
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute
    
    
    def find_best_gain_me(data,attributes,label):
        label_dict = Algorithms.make_variable_dict(data,label)
        set_majority_error = Algorithms.get_majority_error(len(data),label_dict)
        max_gain = 0
        gain_attribute = attributes[0]
        for attribute in attributes:
            gain = set_majority_error
            var_dict = Algorithms.make_variable_dict(data,attribute)
            for key in var_dict.keys():
                subset = Algorithms.data_subset(data,attribute,key)
                subset_dict = Algorithms.make_variable_dict(subset,label)
                gain -= ((var_dict[key]/len(data))*Algorithms.get_majority_error(len(subset),subset_dict))
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute
    
    
    def find_best_gain_g(data,attributes,label):
        label_dict = Algorithms.make_variable_dict(data,label)
        set_gini = Algorithms.get_gini(len(data),label_dict)
        max_gain = 0
        gain_attribute = attributes[0]
        for attribute in attributes:
            gain = set_gini
            var_dict = Algorithms.make_variable_dict(data,attribute)
            for key in var_dict.keys():
                subset = Algorithms.data_subset(data,attribute,key)
                subset_dict = Algorithms.make_variable_dict(subset,label)
                gain -= ((var_dict[key]/len(data))*Algorithms.get_gini(len(subset),subset_dict))
            if gain > max_gain:
                max_gain = gain
                gain_attribute = attribute
        return gain_attribute
    
    
    def build_tree(data,attributes,label,node,depth,split):
        if depth == 1 or len(attributes) == 0:
            node.data = Algorithms.get_most_common(Algorithms.make_variable_dict(data,label))
            return node
        elif Algorithms.label_check(data,label):
            node.data = data[0][label]
            return node
        else:
            if split == 1:
                best_gain = Algorithms.find_best_gain_e(data,attributes,label)
            elif split == 2:
                best_gain = Algorithms.find_best_gain_me(data,attributes,label)
            else:
                best_gain = Algorithms.find_best_gain_g(data,attributes,label)
            node.data = best_gain
            split_dict = Algorithms.make_variable_dict(data,best_gain)
            for key in split_dict.keys():
                data_subset = Algorithms.data_subset(data,best_gain,key)
                newNode = Node(None)
                if len(data_subset) == 0:
                    newNode.data = Algorithms.get_most_common(Algorithms.make_variable_dict(data,label))
                else:
                    attributes.remove(best_gain)
                    Algorithms.build_tree(data_subset,attributes,label,newNode,depth-1,split)
                    attributes.append(best_gain)
                node.children[key] = newNode
            return node
    
    
    def build_tree_array(splits,depths,file):
        tree_array = []
        label = []
        for split in splits:
            for depth in depths:
                data_label = Algorithms.read_file(file)
                attribute_array = Algorithms.make_attributes_array(data_label[1])
                tree_array.append(Algorithms.build_tree(data_label[0],attribute_array,data_label[1],Node(None),depth,split))
                label.append(data_label[1])
        return tree_array,label
    def predict_label(tree,dict,label):
        dict_label = dict[label]
        current_node = tree
        while len(current_node.children) > 0:
            var = current_node.data
            branch = dict[var]
            if branch in current_node.children.keys():
                current_node = current_node.children[branch]
            else:
                return False
        return current_node.data == dict_label
    def get_prediction_errors(var,testing_data):
        i = 0
        return_array = []
        for tree in var[0]:
            error_counter = 0
            label = var[1][i]
            i +=1
            data_length = len(testing_data)
            for dict in testing_data:
                if not Algorithms.predict_label(tree,dict,label):
                    error_counter+=1
            return_array.append(error_counter/data_length)
        return return_array

test_splits = [1,2,3]
split_names = ['information_gain','majority_error', 'gini_index']
test_depths = [1,2,3,4,5,6]
training_file = './car/train.csv'
test_files = ['./car/train.csv','./car/test.csv']

tree_array__label = Algorithms.build_tree_array(test_splits,test_depths,training_file)
testing_data = []
i = 0
for file in test_files:
    testing_data.append(Algorithms.read_file(file)[0])
error_array = []
for data in testing_data:
    error_array.append(Algorithms.get_prediction_errors(tree_array__label,data))
f = open("./Decision_Tree_Data.txt","w")
i = 0
for file in test_files:
    j = 0
    for split in split_names:
        for depth in test_depths:
            f.write("file: "+file +" split: "+split+" depth: "+str(depth)+" error %: "+str(error_array[i][j]) + "\n")
            j+=1
    i+=1
f.close()

