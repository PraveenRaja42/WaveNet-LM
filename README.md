# WaveNet-LM
WaveNet-autoregressive-character-level-Language-Model

Inspired by Andrej Karpathy's lectures Neural Networks: Zero to Hero

A character-level language model using the WaveNet architecture, similar to the one proposed by [DeepMind in 2016](https://www.deepmind.com/research/highlighted-research/wavenet). wavenet_lm.py takes **names.txt** file as input for training. names.txt is a dataset of 32033 names. 

I adapted the WaveNet architecture to the character-level language modeling task, where the goal is to predict the next character given the previous characters. I used a hierarchical structure, as described in the WaveNet paper, to progressively fuse information from different levels of abstraction.
The model consists of three main components:

- An embedding layer that maps each character to a 24-dimensional vector.
- A stack of linear, tanh and batch normalization blocks that process the flattened consecutive embedded characters and produce hidden representations.  
- A softmax layer that outputs a probability distribution over the vocabulary for each character position.
Once trained, the model starts generating realistic and novel names. 
