# Heuristic search algorithm for finding Evidence Counterfactuals based on LIME and lin-SEDC

The LIME-Counterfactual (LIME-C) algorithm is a model-agnostic heuristic search algorithm for finding Evidence Counterfactuals, which are instance-level explanations for explaining model predictions of any classifier. It returns a minimal set of features so that removing these features results in a predicted class change. Removing means setting the corresponding feature value to zero. LIME-Counterfactual has been proposed [in this paper](https://arxiv.org/abs/1912.01819) for explaining binary classifiers trained on high-dimensional, sparse data such as textual data and behavioral big data. 

LIME-C is a hybrid algorithm that makes use of the LIME explainer (proposed by [Ribeiro et al.(2016)](https://dl.acm.org/doi/10.1145/2939672.2939778) for explaining model predictions) and the linear implementation of the SEDC algorithm (proposed [in this paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2282998) as a best-first search algorithm to explain document classifications). The algorithm chooses features to consider as part of the Evidence Counterfactual based on their overall importance for the predicted score. These importance weights can be computed by an additive feature attribution method, such as LIME. The idea is that, the more accurate the importance rankings are, the more likely it is to find a counterfactual explanation starting from removing the top-ranked feature, and so on, up until the predicted class changes. LIME-Counterfactual has shown to have stable effectiveness and efficiency, making it a suitable alternative to SEDC, especially for nonlinear models and instances that are "hard to explain" (i.e., many features need to be removed before the class changes). LIME-Counterfactual immediately solves an issue related to additive feature attribution techniques (such as LIME and SHAP) for high-dimensional data, namely, how many features to show in the explanation? How to set this parameter? For the Evidence Counterfactual the answer is, that number of features such that, when removing them, the predicted class changes. 

At the moment, LIME-C supports binary classifiers built on high-dimensional, sparse data where a "zero" feature value corresponds to the "absence" of the feature. For instance, for behavioral data such as web browsing data, visiting an URL would set the feature value to 1, else 0. The "nonzero" value indicates that the behavior is present or the feature is "active". Setting the feature value to zero would remove this evidence from the browsing history of a user. Another example is textual data, where each token is represented by an individual feature. Setting the feature value (term frequency, tf-idf, etc.) to zero would mean that the corresponding token is removed from the document. Because the reference value when removing a feature from the instance is zero (zero means "missing"), and only active features can be part of the Evidence Counterfactual explanation, the LIME-C implementation makes use of the [LimeTextExplainer](https://github.com/marcotcr/lime/blob/master/lime/lime_text.py). Moreover, LimeTextExplainer uses the cosine distance to measure how "far" the perturbed instances are from the original instance (which makes more sense in the context of big, sparse data than the Euclidean distance that's used in LimeTabular). 
