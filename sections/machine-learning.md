## Machine Learning

### Learning Paradigms
- <a href="https://www.ibm.com/think/topics/supervised-learning" target="_blank">Supervised Learning</a> – Learn from labeled data to make predictions.  
- <a href="https://www.ibm.com/think/topics/unsupervised-learning" target="_blank">Unsupervised Learning</a> – Find patterns in unlabeled data.  
- <a href="https://en.wikipedia.org/wiki/Semi-supervised_learning" target="_blank">Semi-supervised Learning</a> – Combine small labeled data with large unlabeled sets.  
- <a href="https://www.ibm.com/think/topics/self-supervised-learning" target="_blank">Self-supervised Learning</a> – Learn representations without human-labeled data.  
- <a href="https://en.wikipedia.org/wiki/Reinforcement_learning" target="_blank">Reinforcement Learning</a> – Learn through rewards and penalties via trial and error.  
- <a href="https://en.wikipedia.org/wiki/Online_machine_learning" target="_blank">Online Learning</a> – Continuously update the model with new data.  
- <a href="https://en.wikipedia.org/wiki/Transfer_learning" target="_blank">Transfer Learning</a> – Transfer knowledge from one task to another.  
- <a href="https://en.wikipedia.org/wiki/Curriculum_learning" target="_blank">Curriculum Learning</a> – Learn tasks in an easy-to-hard sequence.  
- <a href="https://en.wikipedia.org/wiki/Active_learning_%28machine_learning%29" target="_blank">Active Learning</a> – Selectively query data to improve learning.  
- <a href="https://research.ibm.com/blog/what-is-federated-learning" target="_blank">Federated Learning</a> – Train models across decentralized data sources.  
- <a href="https://www.ibm.com/think/topics/meta-learning" target="_blank">Meta Learning</a> – Learn how to learn across tasks or models.  
- <a href="https://en.wikipedia.org/wiki/Multi-task_learning" target="_blank">Multi-task Learning</a> – Learn multiple tasks with shared representations.  
- <a href="https://en.wikipedia.org/wiki/Few-shot_learning" target="_blank">Few-shot Learning</a> – Learn from a very small number of examples.  
- <a href="https://en.wikipedia.org/wiki/Zero-shot_learning" target="_blank">Zero-shot Learning</a> – Perform tasks with no labeled examples seen.  
- <a href="https://en.wikipedia.org/wiki/Continual_learning" target="_blank">Continual Learning</a> – Retain and build knowledge over time.  
- <a href="https://huggingface.co/learn/nlp-course/chapter5/5?fw=pt" target="_blank">Contrastive Learning</a> – Learn by distinguishing between similar and dissimilar examples.  

### Core Concepts
- <a href="https://learn.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-architecture#models" target="_blank">Model</a> – A trained representation that maps input data to outputs based on learned patterns.
- <a href="https://learn.microsoft.com/en-us/azure/machine-learning/concept-data" target="_blank">Dataset</a> – A collection of data used for training, validating, or testing models.
- <a href="https://en.wikipedia.org/wiki/Training,_validation,_and_test_data" target="_blank">Labeled Data</a> – Data paired with correct output values used in supervised learning.
- <a href="https://en.wikipedia.org/wiki/Feature_(machine_learning)" target="_blank">Feature</a> – An individual measurable property of the data used for making predictions.
- <a href="https://learn.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-architecture#data-labels" target="_blank">Label</a> – The target variable or output that the model aims to predict.
- <a href="https://www.ibm.com/cloud/learn/epoch-machine-learning" target="_blank">Epoch</a> – One complete pass through the entire training dataset.
- <a href="https://en.wikipedia.org/wiki/Batch_normalization" target="_blank">Batch</a> – A subset of training data used in a single iteration of model training.
- <a href="https://en.wikipedia.org/wiki/Loss_function" target="_blank">Loss Function</a> – A function that measures the difference between predicted and actual outputs.
- <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent" target="_blank">Optimizer</a> – Algorithm used to minimize the loss function during training.
- <a href="https://en.wikipedia.org/wiki/Regularization_(mathematics)" target="_blank">Regularization</a> – Techniques to reduce overfitting by penalizing model complexity.
- <a href="https://en.wikipedia.org/wiki/Overfitting" target="_blank">Overfitting</a> – When a model learns the training data too well and performs poorly on new data.
- <a href="https://en.wikipedia.org/wiki/Underfitting" target="_blank">Underfitting</a> – When a model is too simple to capture the patterns in the data.
- <a href="https://en.wikipedia.org/wiki/Bias–variance_tradeoff" target="_blank">Bias-Variance Tradeoff</a> – Balancing error due to bias and variance to improve model performance.
- <a href="https://en.wikipedia.org/wiki/Hyperparameter_optimization" target="_blank">Hyperparameters</a> – Configuration variables set before training that influence model behavior.
- Evaluation Metrics
  - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html" target="_blank">Accuracy</a> – Proportion of correct predictions made by the model.
  - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html" target="_blank">Precision</a> – Proportion of predicted positives that are actually positive.
  - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html" target="_blank">Recall</a> – Proportion of actual positives that are correctly predicted.
  - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html" target="_blank">F1 Score</a> – Harmonic mean of precision and recall.
  - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html" target="_blank">ROC-AUC</a> – Measures a classifier’s ability to distinguish between classes.
- <a href="https://scikit-learn.org/stable/modules/cross_validation.html" target="_blank">Cross-validation</a> – Technique for assessing model performance by splitting data into multiple train-test subsets.

### Algorithms
- <a href="https://scikit-learn.org/stable/modules/linear_model.html#linear-regression" target="_blank">Linear Regression</a> – Predicts a continuous outcome based on the linear relationship between features.
- <a href="https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression" target="_blank">Logistic Regression</a> – Predicts binary or multiclass outcomes using a logistic function.
- <a href="https://scikit-learn.org/stable/modules/tree.html#classification" target="_blank">Decision Tree</a> – A tree-like model used for classification or regression by learning decision rules.
- <a href="https://scikit-learn.org/stable/modules/ensemble.html#random-forests" target="_blank">Random Forest</a> – An ensemble of decision trees that improves accuracy by averaging predictions.
- <a href="https://scikit-learn.org/stable/modules/neighbors.html#classification" target="_blank">k-Nearest Neighbors</a> – Classifies data based on the majority label among its nearest neighbors.
- <a href="https://scikit-learn.org/stable/modules/svm.html" target="_blank">Support Vector Machine (SVM)</a> – Finds the optimal boundary (hyperplane) that best separates classes in the feature space.
- <a href="https://scikit-learn.org/stable/modules/naive_bayes.html" target="_blank">Naive Bayes</a> – Probabilistic classifier based on Bayes’ theorem with strong (naive) independence assumptions.
- <a href="https://xgboost.readthedocs.io/en/stable/" target="_blank">XGBoost</a> – Gradient boosting framework known for speed and performance in structured data tasks.
- <a href="https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting" target="_blank">Gradient Boosting Machines</a> – Builds models sequentially, each one correcting the errors of its predecessor.
- Clustering
  - <a href="https://scikit-learn.org/stable/modules/clustering.html#k-means" target="_blank">K-means</a> – Partitions data into K clusters based on proximity to cluster centroids.
  - <a href="https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering" target="_blank">Hierarchical</a> – Builds nested clusters by successively merging or splitting them.
  - <a href="https://scikit-learn.org/stable/modules/clustering.html#dbscan" target="_blank">DBSCAN</a> – Groups data based on density, identifying clusters of arbitrary shape and handling outliers.
- Dimensionality Reduction
  - <a href="https://scikit-learn.org/stable/modules/decomposition.html#pca" target="_blank">PCA (Principal Component Analysis)</a> – Reduces the dimensionality of data by projecting it onto principal components.
  - <a href="https://scikit-learn.org/stable/modules/manifold.html#t-sne" target="_blank">t-SNE</a> – Non-linear technique for visualizing high-dimensional data in 2 or 3 dimensions.
  - <a href="https://umap-learn.readthedocs.io/en/latest/" target="_blank">UMAP</a> – Fast and scalable dimensionality reduction technique for visualizing structure in data.