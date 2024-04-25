import nltk
from nltk.stem.lancaster import LancasterStemmer
import boto3
import json
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Ensure punkt is downloaded
nltk.download('punkt')

# Initialize stemmer
stemmer = nltk.LancasterStemmer()

# AWS S3 setup
s3 = boto3.client('s3')
bucket_name = 'ai-chatbot-data-nidhi'
object_key = 'intents.json'

# Fetch the data from S3
response = s3.get_object(Bucket=bucket_name, Key=object_key)
content = response['Body'].read().decode('utf-8')
data = json.loads(content)

# Prepare data
words, labels, docs_x, docs_y = [], [], [], []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Data preprocessing
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# Create training data
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = [0 if word not in [stemmer.stem(w.lower()) for w in doc] else 1 for word in words]
    output_row = list(out_empty)
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)


# Model definition
def train_model(X, y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(y[0]), activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(X, y, epochs=200, batch_size=5, verbose=1)
    return model


# Save the trained model
def save_model(model):
    model.save('models/chat_model.h5')


if __name__ == "__main__":
    model = train_model(training, output)
    save_model(model)


def chatbot_response(text):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(text)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    results = model.predict(np.array([bag]))[0]
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        return random.choice(responses)
    else:
        return "I didn't get that. Can you explain or try again?"


print("Start talking with the bot (type quit to stop)!")
while True:
    inp = input("You: ")
    if inp.lower() == "quit":
        break
    response = chatbot_response(inp)
    print(f"Bot: {response}")
