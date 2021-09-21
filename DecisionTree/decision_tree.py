import math
class Node:
    def __init__(self,data):
        self.data = data
        self.children = []
    def print_children(self):
        for child in self.children:
            print(child.data)
            child.print_children()
        return
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

    def find_best_gain(data,attributes,label):
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


    def entropy_tree(data,attributes,label,node,depth):
        if depth == 1:
            node.data = Algorithms.get_most_common(Algorithms.make_variable_dict(data,label))
            return node
        best_gain = Algorithms.find_best_gain(data,attributes,label)
        node.data = best_gain
        split_dict = Algorithms.make_variable_dict(data,best_gain)
        for key in split_dict.keys():
            data_subset = Algorithms.data_subset(data,best_gain,key)
            newNode = Node(None)
            if len(data_subset) == 0:
                newNode.data = Algorithms.get_most_common(Algorithms.make_variable_dict(data,label))
                node.children.append(newNode)
            else:
                attributes.remove(best_gain)
                Algorithms.entropy_tree(data_subset,attributes,label,newNode,depth-1)
                attributes.append(best_gain)
            node.children.append(newNode)
            
        return node
    def majority_error_tree():
        return
    def gini_tree():
        return

    def id3_algorithm(data,attributes,label,split,max_depth):
        if Algorithms.label_check(data,label):
            return Node(True,data[0][label])
        elif len(attributes) == 0:
            label_dict = Algorithms.make_variable_dict(data,label)
            return Node(True,Algorithms.get_most_common(label_dict))
        else:
            if split == 1:
                root_node = Algorithms.entropy_tree(data,attributes,label,Node(None),max_depth)
                print(root_node.data)
                root_node.print_children()
            elif split == 2:
                Algorithms.majority_error_tree()
            else:
                Algorithms.gini_tree()

test_split = 1
test_depth = 6
return_vals = Algorithms.read_file('DecisionTree/car/train.csv')
attribute_array = Algorithms.make_attributes_array(return_vals[1])
Algorithms.id3_algorithm(return_vals[0],attribute_array,return_vals[1],test_split,test_depth)
    

