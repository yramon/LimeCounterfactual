# -*- coding: utf-8 -*-
"""
LIME-Counterfactual
Function for explaining instances with use of counterfactuals.
Step 1: calculate importance-ranked list using LIME.
Step 2: generate counterfactuals from this importance-ranked list.

"""

import lime

classification_model = model

if datatype=="behavioral":
    c=make_pipeline(vectorizer, classification_model) 
    feature_names=vectorizer.get_feature_names()

elif datatype=="text":
    c=make_pipeline(vectorizer_lemma, classification_model) 
    feature_names=vectorizer_lemma.get_feature_names()
   
    
        index=indices_probs_pos[i]
        instance=x_test[index]
        instance=np.reshape(instance,(1,len(feature_names)))
        instance=lil_matrix(instance)
    
#%%
    
def LIME_Counterfactual (instance, classifier_fn, num_features, class_names):
    
    tic=time.time()

    explainer = LimeTextExplainer(class_names = class_names)
    classifier = classifier_fn.predict_proba

    exp = explainer.explain_instance(instance, classifier, num_features = num_features)
    explanation_lime=exp.as_list()
            
    indices_features_lime=[]
    feature_expl_lime=[]
    for j in range(len(explanation_lime)):
        feature=explanation_lime[j][0]
        index_feature=np.argwhere(feature_names==feature)
        indices_features_lime.append(index_feature)
        feature_expl_lime.append(feature)
    feature_coefficient=[]
    for j in range(len(explanation_lime)):
        coefficient=explanation_lime[j][1]
        feature_coefficient.append(coefficient)
        
    size = np.size(feature_expl_lime)
    if (np.size(feature_expl_lime)==0):
        size=np.nan
            
    if (np.size(instance) != 0):
        score=classifier_fn(instance)
        class_instance=classification_model.predict(instance)
        class_new = class_instance
        k=0
        iter=0
        while ((class_new == class_instance) and (k!=len(features))):
            indices_features_explanations_abs_found=[]
            number_perturbed=0
            k+=1
            perturbed_instance=instance.copy()
            j=0
            for feature in features[0:k]:
                if (feature_coefficients[i][j]>=0):
                    number_perturbed+=1
                    if (np.size(feature) != 0):
                        perturbed_instance[:,feature[0][0]]=0
                        indices_features_explanations_abs_found.append(feature[0][0])
                j+=1
            score_new=classifier_fn(perturbed_instance)
            if (score_new[0] < threshold_classifier_probs):
                class_new=class_instance+1
            else: 
                class_new=class_instance
            iter+=1
                
        if (class_new != class_instance):
            toc = time.time()
            time_extra=toc-tic
            explanation_found_lime_abs.append(number_perturbed)
            explanation_found_lime_rel.append(number_perturbed/nb_active_elements_pos[i])
            time_to_find_explanation_lime_found.append(time_to_find_explanation_lime[i] + time_extra)
            difference_score_lime.append(score-score_new)
            nb_active_elements_lime_found.append(nb_active_elements_pos[i])
            indices_explanation_found_lime.append(i)
            expl_lime.append(explanations_lime[i])
                
        else:
            explanation_found_lime_abs.append(np.nan)
            explanation_found_lime_rel.append(np.nan)
            time_to_find_explanation_lime_found.append(np.nan)
            difference_score_lime.append(np.nan)
            nb_active_elements_lime_found.append(nb_active_elements_pos[i])
            indices_explanation_found_lime.append(np.nan)
            expl_lime.append(explanations_lime[i])
    else: 
        explanation_found_lime_abs.append(np.nan)
        explanation_found_lime_rel.append(np.nan)
        time_to_find_explanation_lime_found.append(np.nan)
        difference_score_lime.append(np.nan)
        nb_active_elements_lime_found.append(nb_active_elements_pos[i])
        indices_explanation_found_lime.append(np.nan)
        expl_lime.append(explanations_lime[i])
    i+=1
    indices_features_lime_abs.append(indices_features_explanations_abs_found)
        