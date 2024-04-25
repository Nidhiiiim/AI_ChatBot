# AI_ChatBot
#What are the imports Made?
1) nltk: This is the Natural Language Toolkit, a library used for working with human language data in Python. It provides easy-to-use interfaces to over 50 corpora and lexical resources.
2) LancasterStemmer: A stemming algorithm provided by the NLTK library that reduces words to their root form. For example, "running" would be stemmed to "run".
3) numpy: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
4) random: This module implements pseudo-random number generators for various distributions. It is used here to randomly select a response from a list of possible responses.
5) json: JSON is a lightweight data interchange format inspired by JavaScript object literal syntax. This is used to handle the data (intents and responses) stored in JSON format.

Ensure all below pointers are met to get started!:

Dependencies and NLTK Setup: You've imported necessary libraries and ensured the 'punkt' tokenizer models are downloaded for NLTK.

Stemmer Initialization: You initialize the Lancaster Stemmer for word normalization.

AWS S3 Setup: You set up AWS S3 to fetch the intents.json file which contains data needed for training the bot.

Data Loading and Preprocessing: You load the data from S3, tokenize and stem words, prepare the labels, and generate a bag of words model for each sentence.

Model Training: You define a function to build and train a neural network model using Keras. The model architecture consists of dense layers with dropout for regularization and is trained with SGD optimizer.

Model Saving: You have a function to save the trained model locally.

Chatbot Interaction: Finally, you set up a simple loop to allow real-time chatting with the bot, processing the user's input using the trained model to generate responses.