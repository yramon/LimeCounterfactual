"""
Function to preprocess numerical/binary big data to raw text format and to 
fit the CountVectorizer object to be used in the class object LimeCounterfactual.
"""

"""
Import libraries 
"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np 

class Preprocess_LimeCounterfactual(object):
    def __init__(self, binary_count=True):
        """ Init function
        
        Args:
            binary_count: [“True” or “False”]  when the original data matrix
            contains binary feature values (only 0s and 1s), binary_count is "True". 
            All non-zero counts are set to 1. Default is "True".
        """
        self.binary_count = binary_count
        
    def instance_to_text(self, instance_idx):
        """Function to generate raw text string from instance on (behavioral) big data"""
        active_elements = np.nonzero(instance_idx)[1]
        instance_text=''
        for element in active_elements: 
            instance_text+=" "+'a'+np.str(element)
        return instance_text
    
    def fit_vectorizer(self, instance_idx):
        """Function to fit vectorizer object for (behavioral) big data based on CountVectorizer()"""
        instance_text1=''
        instance_text2=''
        for element in range(np.size(instance_idx.toarray())): 
            instance_text1+=" "+'a'+np.str(element)
            instance_text2+=" "+'a'+np.str(element)
        artificial_text = [instance_text1, instance_text2]
        vectorizer = CountVectorizer(binary = self.binary_count)
        vectorizer.fit_transform(artificial_text)
        feature_names_indices = vectorizer.get_feature_names()
        return vectorizer, feature_names_indices


