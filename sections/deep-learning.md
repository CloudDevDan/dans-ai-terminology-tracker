## Deep Learning

### Architectures
- <a href="https://learn.microsoft.com/en-us/azure/machine-learning/concept-deep-learning-vs-machine-learning" target="_blank">Artificial Neural Networks (ANN)</a> – Basic architecture of interconnected nodes inspired by the human brain.
- <a href="https://en.wikipedia.org/wiki/Convolutional_neural_network" target="_blank">Convolutional Neural Networks (CNN)</a> – Specialized for processing grid-like data such as images using convolutional layers.
- <a href="https://en.wikipedia.org/wiki/Recurrent_neural_network" target="_blank">Recurrent Neural Networks (RNN)</a> – Designed to handle sequential data by maintaining state through cycles.
- <a href="https://en.wikipedia.org/wiki/Long_short-term_memory" target="_blank">Long Short-Term Memory (LSTM)</a> – A type of RNN that solves the vanishing gradient problem with memory cells.
- <a href="https://en.wikipedia.org/wiki/Gated_recurrent_unit" target="_blank">Gated Recurrent Unit (GRU)</a> – A simpler alternative to LSTMs, using gating to retain context in sequences.
- <a href="https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)" target="_blank">Transformers</a> – Foundation of modern LLMs; uses attention mechanisms instead of recurrence.
- <a href="https://en.wikipedia.org/wiki/Autoencoder" target="_blank">Autoencoders</a> – Unsupervised networks that learn to compress and reconstruct input data.
- <a href="https://en.wikipedia.org/wiki/Variational_autoencoder" target="_blank">Variational Autoencoders (VAE)</a> – A probabilistic version of autoencoders used for generative modeling.
- <a href="https://en.wikipedia.org/wiki/Generative_adversarial_network" target="_blank">GANs (Generative Adversarial Networks)</a> – Consist of a generator and a discriminator competing to improve generative output.
- <a href="https://huggingface.co/blog/annotated-diffusion" target="_blank">Diffusion Models</a> – Generative models that learn by reversing a gradual noise process.
- <a href="https://machinelearningmastery.com/the-attention-mechanism-from-scratch/" target="_blank">Attention Mechanism</a> – Allows models to focus on relevant parts of the input when making predictions.

### Components
- <a href="https://en.wikipedia.org/wiki/Artificial_neuron" target="_blank">Neuron</a> – The fundamental unit in a neural network, mimicking a biological neuron.
- Activation Function
  - <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)" target="_blank">ReLU</a> – Outputs zero for negative inputs, otherwise outputs the input itself.
  - <a href="https://en.wikipedia.org/wiki/Sigmoid_function" target="_blank">Sigmoid</a> – Maps input values to a range between 0 and 1.
  - <a href="https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent" target="_blank">Tanh</a> – Maps input values to a range between -1 and 1.
- <a href="https://en.wikipedia.org/wiki/Backpropagation" target="_blank">Backpropagation</a> – Algorithm for training neural networks by propagating errors backward.
- <a href="https://en.wikipedia.org/wiki/Dropout_(neural_networks)" target="_blank">Dropout</a> – Regularization technique where random neurons are ignored during training.
- <a href="https://pytorch.org/docs/stable/nn.init.html" target="_blank">Weight Initialization</a> – Method of assigning initial weights to neurons to improve training convergence.