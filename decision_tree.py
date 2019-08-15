from __future__ import division
import numpy as np
import copy
import unittest

#######################################The decision tree Algorithm######################################
#cpt=0

class DecisionTree:
    def __init__(self,data):
        self.tree = None
        self.data = data
        self.classes = None
        self.attrs = self.get_attrs_data(data)
        pass
        
    def build_tree(self,data):
        '''
        We suppose the following data structure
        Example1: data[0]=[Attr1 Attr2 Attr3......]
        Example2: data[1]=[Attr1 Attr2 Attr3......]
        Example3: data[2]=[Attr1 Attr2 Attr3......]
        .......

        With the last attribute being the observed result

        input: 
            our data

        return: 
            tree
                description: the decision tree
                type: dict
        '''

        #a verification that the recursive solution is working as expected
        #global cpt
        #cpt+=1
        #print(cpt)

        tree = dict()
        attribute, values_wrt_attribute = self.argmax_gain(data)

        if not (attribute in tree):
            tree[attribute] = dict()

        for value in values_wrt_attribute:
            subdata = values_wrt_attribute[value]
            if (len(subdata[0])>1) and (self.entropy(subdata)>0):
                #we split only when we have attributes left to split
                #and that bring more information, i.e entropy non zero
                tree[attribute][value] = self.build_tree(subdata)
            else:
                #we return the decision
                tree[attribute][value] = subdata[0][-1]
                
        return tree

    def argmax_gain(self,data):
        '''
        Takes our data and tries to find the attribute that ensures maximum information gain

        input: 
            our data

        return: 
            col_attr
                description: the chosen attribute on which we execute the split 
                type: string
        return:
            optimal_values_wrt_attribute
                description: the different values that belong to the chosen attribute, to reduce computational efforts
                type: dict
        '''
        #entropy of the whole dataset before performing any splitting
        total_entropy = self.entropy(data)

        max_gain = -np.Inf
        col_attr = None
        optimal_values_wrt_attribute = None

        for attribute in range(len(data[0])-1):
            gain =  total_entropy
            values_wrt_attribute = self.group_data_by_value(data,attribute)

            #the gain after performing splitting
            #with respect to the current attribute
            for value in values_wrt_attribute:
                subdata = values_wrt_attribute[value]
                gain-=np.sum(float(len(subdata)/len(data))*self.entropy(subdata))

            #choose the maximum gain
            #save the best parameters
            #return all calculated variables to avoid calculating them again
            if gain>max_gain:
                max_gain = gain
                optimal_values_wrt_attribute = values_wrt_attribute
                col_attr = self.which_class(list(set(np.array(data)[:,attribute])),self.attrs)

        return col_attr, optimal_values_wrt_attribute

    def entropy(self,subset):
        '''
        Calculates the entropy of a given dataset based on the frequency of every result class

        input:
            our dataset (subset of dataset)
        
        return:
            e
                description: the calculated entropy
                type: float
        
        '''

        #result classes are found in the last column of every dataset or subset
        column = len(subset[0])

        #group our subset of data with respect to the decision, i.e last column in dataset
        group = self.group_data_by_value(subset,column)
        #different decision values
        group_k = group.keys()

        e = 0
        for v in group_k:
            #probability of the decision class
            p = len(group[v])/len(subset)
            e-=(p*np.log(p))

        return e

    def group_data_by_value(self,data, attribute):
        '''
        Divides our dataset with respect to the different values that exist under an attribute

        return:
            group 
                type: dict
                keys: values of the attribute
                items: subset from the dataset sharing the same attribute's value
        '''

        group = dict()
        for i in range(len(data)):
            try:
                val = data[i][attribute]
            except:
                val = data[i][attribute-1]

            #add all rows in the dataset that share the same attribute's value
            #under the same dictionary key
            if val in group:
                group[val].append(data[i][:attribute]+data[i][attribute+1:])
            else:
                group[val] = [data[i][:attribute]+data[i][attribute+1:]]

        return group

    def which_class(self,keys, attrs):
        '''
        Determines the class of the attributes

        return:
            c
                type: string
                description: class that describes the attributes
        '''

        for ind, l in enumerate(attrs):
            check = all(elem in l for elem in keys)
            if check:
                c = self.classes[ind].upper()
                return c

    def get_attrs_data(self,data):
        '''
        Determines the different attributes for every class in the dataset without redundancy

        return:
            attrs
                type: list of lists
                description: possible attributes of every class
        '''

        tmp1 = np.array(data)
        tmp = tmp1[:,:len(data[0])-1]
        attrs = []

        for i in range(tmp.shape[1]):
            tmp2 = list(set(tmp[:,i]))
            attrs.append(tmp2)
        
        return attrs

############################################################# The test class ##################################################################

class TestDT(unittest.TestCase):
    
    def setUp(self):
        self.dataset = [['sunny', 'hot', 'high' , 'weak','no'],
            ['sunny', 'hot', 'high' , 'strong','no'],
            ['overcast', 'hot', 'high' , 'weak','yes'],
            ['rain', 'mild', 'high' , 'weak','yes'],
            ['rain', 'cool', 'normal' , 'weak','yes'],
            ['rain', 'cool', 'normal' , 'strong','no'],
            ['overcast', 'cool', 'normal' , 'strong','yes'],
            ['sunny', 'mild', 'high' , 'weak','no'],
            ['sunny', 'cool', 'normal' , 'weak','yes'],
            ['rain', 'mild', 'normal' , 'weak','yes'],
            ['sunny', 'mild', 'normal' , 'strong','yes'],
            ['overcast', 'mild', 'high' , 'strong','yes'],
            ['overcast', 'hot', 'normal' , 'weak','yes'],
            ['rain', 'mild', 'high' , 'strong','no']]

        self.classes = ['weather','temperature','humidity','wind','decision']

    def test_build(self):
        C = DecisionTree(self.dataset)
        C.classes = self.classes
        self.assertEqual(C.build_tree(self.dataset),{'WEATHER': {'overcast': 'yes', 'sunny': {'HUMIDITY': {'high': 'no', 'normal': 'yes'}}, 'rain': {'WIND': {'strong': 'no', 'weak': 'yes'}}}})


if __name__ == '__main__':
    unittest.main()

