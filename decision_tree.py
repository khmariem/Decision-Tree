from __future__ import division
import numpy as np
import copy

#######################################The decision tree Algorithm######################################
#cpt=0

class DecisionTree:
    def __init__(self):
        self.tree = None
        self.data = None
        pass
        
    def build_tree(self,data):
        '''
        We suppose the following data structure
        Example1: data[0]=[Attr1 Attr2 Attr3......]
        Example2: data[1]=[Attr1 Attr2 Attr3......]
        Example3: data[2]=[Attr1 Attr2 Attr3......]
        .......

        With the last attribute being the observed result/reward

        input: 
            our data

        return: 
            tree
                description: the decision tree algorithm
                type: dict
        '''

        #a verification that the recursive solution is working as expected
        #global cpt
        #cpt+=1
        #print(cpt)

        tree=dict()
        attribute, values_wrt_attribute = self.argmax_gain(data)

        for value in values_wrt_attribute:
            subdata=values_wrt_attribute[value]
            if (len(subdata[0])>1) and (self.entropy(subdata)>0):
                #we split only when we have attributes left to split
                #and that bring more information, i.e entropy non zero
                tree[value]=self.build_tree(subdata)
            else:
                #we return the decision
                tree[value]=subdata[0][-1]
                
        return tree

    def argmax_gain(self,data):
        '''
        This function takes our data and tries to find the attribute that ensures maximum information gain

        input: 
            our data

        return: 
            attribute
                description: the index of the chosen attribute on which we execute the split 
                type: int
        return:
            optimal_values_wrt_attribute
                description: the different values that belong to the chosen attribute, to reduce computational efforts
                type: dict
        '''
        #entropy of the whole dataset before performing any splitting
        total_entropy = self.entropy(data)

        max_gain=-np.Inf
        column = None
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
                column = attribute
                optimal_values_wrt_attribute = values_wrt_attribute

        return column, optimal_values_wrt_attribute

    def entropy(self,subset):
        '''
        This function calculates the entropy of a given dataset based on the frequency of every result class

        input:
            our dataset (subset of dataset)
        
        return:
            e
                description: the calculated entropy
                type: float
        
        '''

        #result classes are found in the last column of every dataset or subset
        column=len(subset[0])

        #group our subset of data with respect to the decision, i.e last column in dataset
        group = self.group_data_by_value(subset,column)
        #different decision values
        group_k=group.keys()

        e = 0
        for v in group_k:
            #probability of the decision class
            p = len(group[v])/len(subset)
            e-=(p*np.log(p))

        return e

    def group_data_by_value(self,data, attribute):
        '''
        Divides our dataset with respect to the different values that exist under an attribute

        return: type: dict
                keys: values of the attribute
                items: subset from the dataset sharing the same attribute's value
        '''

        group=dict()
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
                group[val]=[data[i][:attribute]+data[i][attribute+1:]]

        return group
