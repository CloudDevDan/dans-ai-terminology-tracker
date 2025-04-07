---
title: Dan’s AI Terminology Tracker
markmap:
  colorFreezeLevel: 2
  activeNode:
    placement: center
  spacingVertical: 8
  initialExpandLevel: 3
---

# Artificial Intelligence

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

## Deep Learning

### Architectures
- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Transformers
- Autoencoders
- Variational Autoencoders (VAE)
- GANs (Generative Adversarial Networks)
- Diffusion Models
- Attention Mechanism

### Components
- Neuron
- Activation Function
  - ReLU
  - Sigmoid
  - Tanh
- Backpropagation
- Dropout
- Weight Initialization

## Generative AI

### Models
- Large Language Model (LLM)
- Small Language Model (SLM)
- Foundation Model
- Multimodal Model
- Fine-tuned Model
- Pretrained Model

### Techniques
- Prompt
- Prompt Engineering
- Chain-of-Thought Prompting
- Instruction Tuning
- Retrieval-Augmented Generation (RAG)
- In-context Learning
- Few-shot Learning – Commonly used during inference for generalization with limited examples.  
- Zero-shot Learning – Allows models to perform tasks they weren't explicitly trained on.  
- Temperature
- Top-k / Top-p Sampling
- Beam Search

### Concepts
- Token
- Tokenization
- Embedding
- Vector Store
- Semantic Search
- Latency
- Throughput
- Hallucination
- Grounding
- Guardrails
- Context Window
- Knowledge Injection

## MLOps & GenAIOps

### MLOps Practices
- Model Lifecycle
  - Training
  - Evaluation
  - Deployment
  - Monitoring
- Model Registry
- Model Versioning
- Model Drift
- Data Drift
- Automated Retraining
- Pipeline Orchestration

### GenAIOps Additions
- Prompt Testing
- Prompt Versioning
- LLM Evaluation
- Human-in-the-Loop (HITL)
- Synthetic Data
- Agent Behavior Monitoring

### GenAIOps Tooling
- <a href="https://github.com/microsoft/prompty" target="_blank">Prompty</a> – Microsoft tool for prompt observability, understandability, and DevOps pipeline integration.
- PromptFoo – Prompt testing and evaluation framework.
- LangSmith – Tracing and analytics for LLM chains.
- HoneyHive – Human feedback for LLM evaluation.
- TruLens – Trace and evaluate LLM behavior in real-time.
- LLM Guard – Add safety filters and constraints to GenAI apps.

### Infrastructure
- Inference Pipeline
- Batch Inference
- Real-time Inference
- Containerization (Docker)
- Kubernetes
- CI/CD for ML

## Data & Pipelines

### Data Types
- Structured Data
- Unstructured Data
- Tabular Data
- Time-Series Data
- Text
- Audio
- Image
- Video

### Processing Steps
- Data Collection
- Data Labeling
- Data Cleaning
- Data Augmentation
- Data Preprocessing
- Feature Engineering
- Feature Selection
- Data Splitting
  - Train
  - Validation
  - Test

### Storage & Access
- Data Lake
- Data Warehouse
- Feature Store
- Vector Database

## Tools & Frameworks

### Programming & Libraries
- Python
- R
- NumPy
- pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- PyTorch
- JAX
- ONNX

### Experimentation
- MLflow
- Weights & Biases
- DVC (Data Version Control)
- Optuna

### Deployment
- FastAPI
- Flask
- Streamlit
- Gradio
- Triton Inference Server

### LLM Toolchains
- LangChain
- LlamaIndex
- PromptFlow
- Semantic Kernel

## Microsoft AI Stack

### Platforms
- <a href="https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning" target="_blank">Azure Machine Learning</a> – Cloud platform for building, training, and deploying ML models at scale.
- <a href="https://azure.microsoft.com/en-us/products/ai-foundry/" target="_blank">Azure AI Foundry</a> – End-to-end platform to accelerate the development of enterprise-ready, AI-powered applications.
- <a href="https://github.com/marketplace?type=actions&query=model" target="_blank">GitHub Model Marketplace</a> – Discover and integrate open models and tools into GitHub workflows.

### Services
- <a href="https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview" target="_blank">Azure OpenAI</a> – Perform a wide variety of natural language tasks using OpenAI models hosted on Azure.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/agents/overview" target="_blank">Azure AI Agent Service</a> – Develop AI agents to automate and execute business processes.
- <a href="https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search" target="_blank">Azure AI Search</a> – Bring AI-powered search to mobile and web applications.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview" target="_blank">Content Safety</a> – An AI service that detects and filters unwanted or harmful content.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/speech-service/overview" target="_blank">Speech</a> – Convert speech to text, text to speech, and recognize speakers.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview" target="_blank">Document Intelligence</a> – Extract structured data from documents and forms.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/overview" target="_blank">Vision</a> – Analyze content in images and videos.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/overview" target="_blank">Custom Vision</a> – Customize image recognition for specific business needs.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/overview-identity" target="_blank">Face</a> – Detect and identify people and emotions in images.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/translator/overview" target="_blank">Translator</a> – Translate text across 100+ languages and dialects.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/language-service/overview" target="_blank">Language</a> – Add industry-leading NLU (natural language understanding) capabilities to your app.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/overview" target="_blank">Content Understanding</a> – Extract structured insights from multimodal content (text, images, etc.).
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/immersive-reader/overview" target="_blank">Immersive Reader</a> – Help users read and comprehend text more effectively.

### Products
- <a href="https://www.microsoft.com/en-us/microsoft-copilot" target="_blank">Microsoft Copilot</a> – Embedded AI assistant across Microsoft 365 apps like Word, Excel, and Outlook.
- <a href="https://learn.microsoft.com/en-us/microsoft-copilot-studio/overview" target="_blank">Microsoft Copilot Studio</a> – Low-code platform for building custom AI copilots integrated with enterprise data.
- <a href="https://learn.microsoft.com/en-us/ai-builder/overview" target="_blank">AI Builder</a> – Add AI to Power Apps and Power Automate using prebuilt or custom models.
- <a href="https://learn.microsoft.com/en-us/power-virtual-agents/fundamentals-what-is-power-virtual-agents" target="_blank">Power Virtual Agents</a> – Build chatbots without code using a guided interface and integrated AI.

### Microsoft Learning Resources
- <a href="https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-fundamentals/" target="_blank">Azure AI Fundamentals (AI-900)</a> – Learn core AI and ML concepts and how they’re implemented in Azure.
- <a href="https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-engineer/" target="_blank">Azure AI Engineer Associate</a> – Explore the skills and training required to pass the AI-102 certification.
- <a href="https://azure.github.io/ai-app-templates/" target="_blank">Build with AI</a> – Get started with AI app templates. Edit and deploy directly to Azure using VS Code or GitHub Codespaces.  
- <a href="https://ai.azure.com/labs" target="_blank">Azure AI Foundry Labs</a> – Explore the latest AI innovations and accelerate your Azure AI journey.
- <a href="https://learn.microsoft.com/en-us/azure/architecture/browse/?terms=AI" target="_blank">Azure AI Reference Architectures</a> – Microsoft’s blueprint scenarios for deploying AI solutions on Azure.

## Ethics, Safety & Governance

### Responsible AI Principles
- Fairness
- Accountability
- Transparency
- Explainability
- Privacy
- Reliability

### Practices
- Model Interpretability
- Bias Mitigation
- Adversarial Robustness
- Safety Guardrails
- Differential Privacy
- Federated Privacy
- AI Governance

### Responsible AI Toolkits
- Fairlearn
- InterpretML
- Responsible AI Dashboard
- OpenAI Eval
- IBM AI Fairness 360

### Core Concepts
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/agents/overview" target="_blank">Azure AI Agent Service</a> – Platform for creating autonomous agents that can reason, plan, and act using plugins and tools.
- <a href="https://devblogs.microsoft.com/foundry/integrating-azure-ai-agents-mcp/" target="_blank">Model Context Protocol (MCP)</a> – An open standard designed to seamlessly connect AI assistants with diverse data sources.
- <a href="https://azure.microsoft.com/en-us/blog/announcing-the-responses-api-and-computer-using-agent-in-azure-ai-foundry/" target="_blank">Computer-Using Agent (CUA)</a> – Agents that operate software interfaces to automate complex tasks.

## Infrastructure & Deployment

### Integration Patterns
- <a href="https://learn.microsoft.com/en-us/azure/app-service/tutorial-sidecar-local-small-language-model" target="_blank">App Service Sidecar Extensions</a> – Extend Azure App Services by attaching GenAI or inference workloads as sidecars.
- <a href="https://learn.microsoft.com/en-us/azure/architecture/ai-ml/architecture/azure-openai-baseline-landing-zone" target="_blank">Azure OpenAI Landing Zone</a> – Reference implementation for secure and scalable OpenAI integration on Azure.

## GenAI Platform Features

### Advanced Capabilities
- <a href="https://azure.microsoft.com/en-us/blog/announcing-the-responses-api-and-computer-using-agent-in-azure-ai-foundry/" target="_blank">Responses API</a> – Observe and analyze LLM output responses for quality, safety, and grounding.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/provisioned-throughput?tabs=global-ptum" target="_blank">Provisioned Throughput Units (PTU)</a> – A model deployment type that allows you to specify the amount of throughput you require in a model deployment.
- <a href="https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/spillover-traffic-management" target="_blank">Provisioned Throughput Units (PTU) Spillover</a> – Control and scale LLM capacity in Azure AI, with automatic spillover on demand.

## Ethics, Safety & Governance

### Governance
- <a href="https://artificialintelligenceact.eu/" target="_blank">EU AI Act</a> – European Union's legislation on trustworthy AI development and deployment.
- <a href="https://www.gov.uk/government/publications/ai-regulation-a-pro-innovation-approach" target="_blank">UK AI Action Plan</a> – UK government’s policy for responsible AI innovation.
- <a href="https://learn.microsoft.com/en-us/azure/defender-for-cloud/ai-threat-protection" target="_blank">Microsoft Defender for AI</a> – Protect and monitor AI models from threats, abuse, and adversarial attacks.

## Tools & Frameworks

### Model Hubs & Registries
- <a href="https://github.com/marketplace?type=actions&query=model" target="_blank">GitHub Model Marketplace</a> – Discover and integrate open models and tools into GitHub workflows.

## Related Fields

### Natural Language Processing (NLP)
- Named Entity Recognition (NER)
- Sentiment Analysis
- Text Classification
- Question Answering
- Summarization
- Translation

### Computer Vision (CV)
- Object Detection
- Image Classification
- Segmentation
- Face Recognition

### Speech
- Text-to-Speech (TTS)
- Speech-to-Text (STT)
- Voice Cloning
- Speaker Identification

### Robotics & Agents
- Embodied AI
- Autonomous Agents
- Digital Twins