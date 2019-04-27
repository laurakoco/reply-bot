import nltk
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle
import re
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

from classes_dict import *

my_words = []
my_classes = []
my_doc = []
ignore_words = ['?']

# pre-process text from classes_dict
for some_class in classes_dict:
    
    my_classes.append(some_class)
    
    for some_pattern in classes_dict[some_class]["pattern"]:
        
        temp_words = []
        
        raw_words = some_pattern
        # raw_words = ' '.join(raw_words)
        word_tokens = nltk.word_tokenize(raw_words)
        
        for some_word in word_tokens:
            if some_word not in ignore_words:
                stemmed_word = stemmer.stem(some_word.lower())
                my_words.append(stemmed_word)
                temp_words.append(stemmed_word)

    my_doc.append((temp_words, some_class))

my_words = sorted( list(set(my_words)) ) # remove duplicate words
my_classes = sorted( list(set(my_classes)) )

training = []
output = []
output_empty = [0] * len(my_classes)

for some_doc in my_doc:
    
    bag = []
    pattern_words = some_doc[0]
    
    # create bag of words array
    for some_word in my_words:
        if some_word in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

output_row = list(output_empty)
output_row[my_classes.index(some_doc[1])] = 1

training.append([bag, output_row])

# shuffle training data and put into array
random.shuffle(training)
training = np.array(training)

# create train lists
train_x = list(training[:,0])
train_y = list(training[:,1])

# reset underlying graph data
tf.reset_default_graph()

# build nn model
net = tflearn.input_data(shape = [None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
net = tflearn.regression(net)

# define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')

# train model
# model.fit(train_x, train_y, n_epoch = 500, batch_size = 8, show_metric = True)
# model.save('model_stupid.tflearn')
# pickle.dump( {'my_words':my_words, 'my_classes':my_classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )

# load model
model.load('./model_laura.tflearn')

# import pickle
# data = pickle.load( open( "training_data", "rb" ) )
# words = data['words']
# classes = data['classes']
# train_x = data['train_x']
# train_y = data['train_y']

def tokenize_input(sentence):
    
    sentence_words = nltk.word_tokenize(sentence) # tokenize pattern
    # sentence_words = [stemmer.stem(word.lower()) for word in sentence_words] # stem each word
    
    for some_word in sentence_words:
        sentence_words[sentence_words.index(some_word)] = stemmer.stem(some_word.lower())
    
    return sentence_words

def bag(user_input):
    
    input_words = tokenize_input(user_input)
    
    bag = [0] * len(my_words)
    
    for input_word in input_words:
        for i in range(0,len(my_words)):
            bag_word = my_words[i]
            if input_word == bag_word:
                bag[i] = 1

return(np.array(bag))

error_threshold = 0.95

def classify(user_input):
    
    results = model.predict([bag(user_input)])[0] # generate probabilities from the model
    
    filtered_results = []
    
    for i in range(0, len(results)):
        this_result = results[i]
        if this_result > error_threshold:
            filtered_results.append([i, this_result])

filtered_results.sort(key = lambda x: x[1], reverse = True)

return_list = []

for i in range(0, len(filtered_results)):
    return_list.append((my_classes[filtered_results[i][0]], filtered_results[i][1]))
    
    return return_list # return tuple of intent and probability

print("* Hello! Type in a message and I will suggest some replies! If you'd like to exit please type quit!")

flag = True

while flag:
    
    user_input = raw_input('>>> ').lower() # get input and convert to lowercase
    
    if not re.search('quit', user_input):
        
        some_array = []
        
        some_array = classify(user_input)
        
        # print "predicted class: " + my_classes[pred_class]
        # print "probability: " + str(pred_prob)
        
        if len(some_array) != 0:
            
            # print "predicted class: " + str(some_array[0][0])
            # print "probability: " + str(some_array[0][1])
            
            for response in classes_dict[some_array[0][0]]["response"]:
                print "* " + response
    
        else:
            
            print "[No Suggestion]"

else:
    
    flag = False

