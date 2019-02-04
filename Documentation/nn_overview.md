# A Soft Introduction to ML and Neural Nets #

## Prelude ##

In this document I will provide a conceptual overview of machine learning and neural networks (which
includes deep learning), provide links to sources, and describe the tools you
will need to begin classifying data with neural networks and dive into the model that I've developed. In the first section, you will learn the basic concepts underpinning the field of machine learning. In the second section, you will be introduced to Neural Networks. At the end of the document, I link to other sources that I found helpful, and that would be good jumping off points for your own exploration.

Once you are familiar with the conceptual underpinnings of Machine Learning and Neural Networks, and would like to know how I constructed the model in this repo, I would check out my more [practical guide](training.md). There, you will learn how to start with poorly structured data, and end up with a clean dataset and a working model.

---

## Machine Learning ##

Machine learning is a subset of field of Artificial Intelligence "that often uses statistical techniques to give computers the ability to "learn" (i.e., progressively improve performance on a specific task) with data, without being explicitly programmed" (from [Wikipedia][ml wiki]). Any Machine Learning solution has three parts: the **Model** (sometimes referred to as the Hypothesis Space), the **Loss Function**, and the **Optimizer**.

The Model (Hypothesis Space) is the set of possible functions. At its essence, the model describes a particular type of ways to map inputs (data) to outputs (labels). 

Consider one of the more intuitive types of Models: [Decision Trees][decision tree]. A Decision Tree can be conceptualized as a series of "If this, then that" statements. The Hypothesis Space of decision trees is all ways to make decisions using "If this, then that". For example, I want to decide if I should wear a rain coat, and I only know one thing: if it is raining or not. Then, the hypothesis space for a Decision Tree on this particular data (raining or not), is "If raining, bring raincoat. If not raining, bring raincoat", "If raining, don't bring raincoat. If not raining, bring raincoat", etc. We are trying to choose the best particular element within the space (which I will refer to as a model, lowercase): in this case, it would be "If raining, bring raincoat. If not raining, don't bring raincoat".

I want to emphasize that the word "Model" can be used in multiple ways in Machine Learning. It may refer to the entire Hypothesis Space we are exploring (we are trying to choose the best function of the structure "If, then"), or it may refer to the specific solution we chose ("If rain, bring raincoat"). From this point on I refer to the entire space as the Model, and specific solutions within the space as a model. Examples of Models include: Decision Trees, Linear Regression, and Neural Networks. Examples of models include: "If rain, bring raincoat", $y = 2x + 4$, and the classifier I built in this repo. 

The Loss function is how we evaluate the performance of a specific model (an element of the Hypothesis Space). In particular, we want a model with the lowest possible loss. Imagine a situation where we are trying to classify photos of Dogs and Cats. We could imagine a very simple loss function: for every datapoint that our model gets wrong, the loss increases by one. Here, the loss of a model is the number of examples it misclassifies.

Once we have defined a Model (a Hypothesis Space) and a Loss Function (a Scorer), we need an algorithmic way to choose the model (element of the Hypothesis Space) with the lowest loss. The whole point of Machine Learning is that the computer can choose the best model on its own! This problem is not as easy as it looks - naive search of the Hypothesis Space (trying every possible element) can either take an exponential amount of time (in the case of Decision Trees), or could even be impossible (there are an infinite number of ways to write a line $y = mx + b$). 

Unfortunately, it is usually infeasible to algorithmically find the *best* possible model in a reasonable amount of time (a so called Global Maximum). Instead, we have to settle for **locally optimal** search algorithms that can at least find [local maxima][maxima]. There are many different version of optimization algorithms. Many of these algorithms (as well as the one used to tune Neural Networks: Backpropagation) rely on an algorithm called **gradient descent**. 

A gradient is a vector that represents the direction of the maximum rate of change in an n-dimentional function. In gradient descent, we go in the opposite direction of the gradient. That is, we follow the directions of steepest descent. Keep in mind: we are trying to minimize a loss function, so it makes sense for us to try to take actions to decrease its value as quickly as possible. This may be visualized like ball rolling down into a bowl, as shown below.

![Gradient Descent](Photos/gradient_descent_1.png)

Note that there are multiples "bowls", or local minima in the function modeled above. Gradient descent guarantees that we will end up in one of those bowls, not that we will end up in the deepest one. We are guaranteed a local minimum, no the global minimum.

Gradient descent is parameterized by its learning rate - how far you follow the direction of steepest descent each step. If your learning rate is too large, you may skip right over the minimum and never "converge" (convergence is like a ball getting stuck at the bottom of a bowl). However, if your learning rate is too small, you will take an incredibly long time to converge. Imagine that you're looking for your exit on a highway. If you go too fast, you may blow right past it, but if you're going to slowly, it will take you forever to get there. The goal is to find a balance.

![Learning Rate](Photos/learning_rate.png)

Backpropagation, the optimization algorithm used to train neural networks, is just a slightly fancier way to do gradient descent. It used to be very hard to calculate gradients on neural networks. The gradient of the earliest nodes (closest to the input) depended on the gradient of the nodes ahead of them, and which depended on the nodes ahead of them, all the way up to the output nodes. The insight of Backpropagation was that we can efficiently calculate the gradient of every nodes using a dynamic programming approach. We calculate the gradient of the output nodes, use that gradient to calculate the gradients of the next layer behind that, and continue the process until we have reached the input layer, tweaking every set of weights a little along the way. If you didn't understand that, don't worry! The specifics aren't critical. If you would like a more mathematical explanation of gradient descent I would try [here][backprop].

### Types of Learning ###

There are three main types of Machine Learning: supervised, unsupervised, and reinforcement. Supervised learning uses labeled input output pairs. Examples: Naive Bayes, Decision Trees, Neural Networks.

Unsupervised learning has no labels and learns patterns inherent in the data. Examples: K Nearest Neighbor, K Clustering.

Reinforcement learning involves an algorithm that attempts to maximize a reward function within an environment. It takes actions, observes the resulting reward, and then modulates its future behavior based on that outcome. For example, a computer could be playing an Atari video game, and every action it makes could earn it points. Over a long sequence of act / observe cycles, it will learn how best to act to accrue the most points. This is how Google DeepMind trained its famous [Atari playing][atari] (and eventually [Go playing][go]) Neural Nets.

As you may have noticed in my description of reinforcement learning, the distinction between supervised and unsupervised learning isn't as strict as I made it out to be. In the Atari example: is it supervised or unsupervised? You could argue either way. We didn't give the model already played games to train on, but it is receiving labeled points from the environment in regards to action / reward pairs. 

Although we like to create clean categories, they tend to bleed into each other. Human learning is like this as well - partly reinforcement, partly supervised. Our parents tell us things (supervised learning) and then we go into the environment and play (reinforcement learning). We often see play as a sort of unsupervised learning. Just keep in mind that in practice, these distinctions may blur.

### Overfitting ###

Avoiding overfitting is one of the most important tasks in any ML algorithm. We can avoid overfitting with regularization, among other things.

---

## Neural Networks ##

Hopefully, you are now comfortable with the three pillars of machine learning - hypothesis spaces (aka models), loss functions, and optimizers. Remember, all any machine learning algorithm consists of in choosing a Model, a Loss Function, and an Optimzer.

Now we are going to describe a particular class of hypothesis spaces - Neural Networks! These are the exact structures that you have heard so much about! There are several variations, but every neural network is built upon the same set of primitives. By the end of this section, you will understand the building blocks of neural networks, the prominent types of networks, and what problems to apply them to.

A neural network is a specialized version of a [graph][graphs]. A graph is made up of things (called nodes) and the connections between those
things (called edges). A neural network is just a graph with a special structure!

In a neural network, the nodes take inputs from their incoming connections,
compute a function (called the ["Activation Function"][activation]), and then pass their
output to the next set of nodes. Eventually, the last set of nodes generates an
output which represents the prediction. This could be a vector (for example, if you want to
predict a probability distribution over a set of classes), or a single number (for example,
if you were trying to predict housing prices). I highly recommend clicking the "Activation Function" link above to learn more about them - they are a critical part of neural nets.

A neural network consists of layers of nodes. The simplest type of Neural Net (called a Feedforward Neural Net or a Multilayer Perceptron) looks like this:

![Multilayer Perceptron](Photos/deep_neural_network.png)

A network has three types of layers: the input layer, hidden layers, and output layers. In the picture they are yellow, blue, and red respectively. The depth of a network refers to the number of layers (tweaked by adding hidden layers) and the width of a network refers roughly to the number of nodes in each layer (I say roughly because, as the picture shows, not every layer needs to have the same number of node). When someone talks about "Deep Learning" they are talking about Deep Neural Networks: networks with a large number of hidden layers.

In order to predict the network first receieves inputs at the input layer where the nodes calculate their respective activation functions on the input (activation function are pretty much always consistent across all nodes in a layer). They then pass their outputs to the next layer, which calculates its set of activations. This process continues until the series of activations passes all the way to the output layer, where it outputs the final series of activations as the final output.

You might be wondering, why does layer size and layer width matter? Keep in mind what we are trying to do: we are trying to model the input / output function inherent in the data. That function may be very complex - that's why we're using a neural net! It happens that wider and deeper neural nets can represent more complex functions.

Highly expressive network come with both benefits and drawbacks. If the network isn't expressive enough, it won't be able to capture the underlying function, and will never learn enough patterns in the training data to generalize. This is called "underfitting". If the network is to expressive it is at risk of overfitting the data. As previously mentioned, the more complex the model, the more data that is needed to train it. Without enough data, the an overly expressive model will fit exactly to the noise in the training data and fail to generalize. Part of the recent explosion in Deep Learning is the explosion of the size of data sets, as well as the necessary computing power to train networks on data sets of that scale.

### Regularization in Neural Nets ###

Dropout. Early stopping. 
 
## Tools ##

ML: Numpy, Scikit-learn, Pandas, Matplotlib.pyplot

NN: Theano, Tensorflow, Keras

## Reading Materials ##

* Blogs
  * [Get walked through real world projects](https://www.kaggle.com/kernels)
  * [Blog of a Practitioner](https://ujjwalkarn.me/)


* Books

  * Machine Learning: [My book from school](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

  * Neural Networks and Deep Learning:
  [Deep Learning Mathematical](http://www.deeplearningbook.org/), [Deep Learning with Python](http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf)

  * Tools: [Getting started with Numpy](http://cs231n.github.io/python-numpy-tutorial/)

<!-- Links! -->
[graphs]: https://en.wikipedia.org/wiki/Graph_theory "Graph theory wiki"

[activation]: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6  "Activation Functions Explained"

[ml wiki]: https://en.wikipedia.org/wiki/Machine_learning

[decision tree]: https://heartbeat.fritz.ai/introduction-to-decision-tree-learning-cd604f85e236

[maxima]: https://www.mathsisfun.com/algebra/functions-maxima-minima.html

[backprop]: http://neuralnetworksanddeeplearning.com/chap2.html

[atari]: https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning/

[go]: https://deepmind.com/blog/alphago-zero-learning-scratch/