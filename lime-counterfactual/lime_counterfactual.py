"""
Function for explaining classified instances using evidence counterfactuals.
"""

"""
Import libraries 
"""
from lime.lime_text import LimeTextExplainer
import time
import numpy as np
    
class LimeCounterfactual(object):
    """Class for generating evidence counterfactuals for classifiers on behavioral/text data"""
    
    def __init__(self, c_fn, classifier_fn, vectorizer, threshold_classifier, 
                 feature_names_full, max_features=30, class_names = ['1','0'], 
                 time_maximum=120):
        
        """ Init function
        
        Args:
            c_fn: [pipeline] for example:
        
                c = make_pipeline(vectorizer, classification_model)
                (where classification_model is a fitted scikit learn model and
                vectorizer is a fitted object) 
            
            classifier_fn: [function] classifier prediction probability function
            or decision function. For ScikitClassifiers, this is classifier.predict_proba 
            or classifier.decision_function or classifier.predict_log_proba.
            Make sure the function only returns one (float) value. For instance, if you
            use a ScikitClassifier, transform the classifier.predict_proba as follows:
                
                def classifier_fn(X):
                    c=classification_model.predict_proba(X)
                    y_predicted_proba=c[:,1]
                    return y_predicted_proba
            
            max_features: [int] maximum number of features allowed in the explanation(s).
            Default is set to 30.
            
            class_names: [list of string values] 
            
            vectorizer: [fitted object] a fitted vectorizer object
            
            threshold_classifier: [float] the threshold that is used for classifying 
            instances as positive or not. When score or probability exceeds the 
            threshold value, then the instance is predicted as positive. 
            We have no default value, because it is important the user decides 
            a good value for the threshold. 
            
            feature_names_full: [numpy.array] contains the interpretable feature names, 
            such as the words themselves in case of document classification or the names 
            of visited URLs.
            
            time_maximum: [int] maximum time allowed to generate explanations,
            expressed in minutes. Default is set to 2 minutes (120 seconds).
        """
        
        self.c_fn = c_fn
        self.classifier_fn = classifier_fn
        self.class_names = class_names
        self.max_features = max_features
        self.vectorizer = vectorizer
        self.threshold_classifier = threshold_classifier
        self.feature_names_full = feature_names_full
        self.time_maximum = time_maximum 
    
    def explanation(self, instance):
        """ Generates evidence counterfactual explanation for the instance.
        
        Args:
            instance: [raw text string] instance to explain as a string
            with raw text in it
                        
        Returns:
            A dictionary where:
                
                explanation_set: features in counterfactual explanation.
                
                feature_coefficient_set: corresponding importance weights 
                of the features in counterfactual explanation.
                
                number_active_elements: number of active elements of 
                the instance of interest.
                                
                minimum_size_explanation: number of features in the explanation.
                
                minimum_size_explanation_rel: relative size of the explanation
                (size divided by number of active elements of the instance).
                
                time_elapsed: number of seconds passed to generate explanation.
                
                score_predicted[0]: predicted score/probability for instance.
                
                score_new[0]: predicted score/probability for instance when
                removing the features in the explanation set (~setting feature
                values to zero).
                
                difference_scores: difference in predicted score/probability
                before and after removing features in the explanation.
                
                expl_lime: original explanation using LIME (all active features
                with corresponding importance weights)
        """
        
        tic = time.time() #start timer
        
        instance_sparse = self.vectorizer.transform([instance])
        nb_active_features = np.size(instance_sparse)
        score_predicted = self.classifier_fn(instance_sparse)
        explainer = LimeTextExplainer(class_names = self.class_names)
        
        classifier = self.c_fn.predict_proba
        
        exp = explainer.explain_instance(instance, classifier, num_features=nb_active_features)
        explanation_lime = exp.as_list()
        
        """
        indices_features_lime = []
        feature_coefficient = []
        feature_names_full_index = []
        for j in range(len(explanation_lime)):
            if explanation_lime[j][1] >= 0:   #only the features with a zero or positive estimated importance weight are considered
                feature = explanation_lime[j][0]
                index_feature = np.argwhere(np.array(self.feature_names_full)==feature) #returns index in feature_names where == feature
                feature_names_full_index.append(self.feature_names_full[index_feature[0][0]])
                indices_features_lime.append(index_feature[0][0])
                feature_coefficient.append(explanation_lime[j][1])
        """         
        if (np.size(instance) != 0):
            score_new = score_predicted
            k = 0
            number_perturbed = 0
            while ((score_new[0] >= self.threshold_classifier) and (k != len(explanation_lime)) and (time.time()-tic <= self.time_maximum) and (number_perturbed < self.max_features)):
                number_perturbed = 0
                feature_names_full_index = []
                feature_coefficient = []
                k += 1
                perturbed_instance = instance_sparse.copy()
                for feature in explanation_lime[0:k]:
                    if feature[1] > 0:
                        index_feature = np.argwhere(np.array(self.feature_names_full)==feature[0])
                        number_perturbed += 1
                        if (len(index_feature) != 0):
                            index_feature = index_feature[0][0]
                            perturbed_instance[:,index_feature] = 0
                            feature_names_full_index.append(index_feature)
                            feature_coefficient.append(feature[1])
                score_new = self.classifier_fn(perturbed_instance)
                    
            if (score_new[0] < self.threshold_classifier):
                time_elapsed = time.time() - tic
                minimum_size_explanation = number_perturbed
                minimum_size_explanation_rel = number_perturbed/nb_active_features
                difference_scores = (score_predicted - score_new)
                number_active_elements = nb_active_features
                expl_lime = explanation_lime
                explanation_set = feature_names_full_index[0:number_perturbed]
                feature_coefficient_set = feature_coefficient[0:number_perturbed]
                
            else:
                minimum_size_explanation = np.nan
                minimum_size_explanation_rel = np.nan
                time_elapsed = np.nan
                difference_scores = np.nan
                number_active_elements = nb_active_features
                expl_lime = explanation_lime
                explanation_set = []
                feature_coefficient_set = []
                
        else: 
            minimum_size_explanation = np.nan
            minimum_size_explanation_rel = np.nan
            time_elapsed = np.nan
            difference_scores = np.nan
            number_active_elements = nb_active_features
            expl_lime = explanation_lime
            explanation_set = []
            feature_coefficient_set = []
            
            
        return {'explanation_set':explanation_set, 'feature_coefficient_set':feature_coefficient_set, 'number_active_elements':number_active_elements, 'size explanation': minimum_size_explanation, 'relative size explanation':minimum_size_explanation_rel, 'time elapsed':time_elapsed, 'original score': score_predicted[0], 'new score':score_new[0], 'difference scores':difference_scores, 'explanation LIME':expl_lime}      