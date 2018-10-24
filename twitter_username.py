"""
Script for the main simulation
based on: https://www.tensorflow.org/tutorials/sequences/text_generation
"""
import json
import os
import time

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from utils import get_tweets_from_screen_name, tweets_preprocessing
from rnn_model import GRU_Model as Model
from rnn_model import split_input_target, loss_function

################################################################################
# Load parameters
with open('params.json',"rt") as fin:
    p = json.load(fin)

# data related stuff
TWITTER_SCREEN_NAME = p['screen_name']
CREDENTIALS_FILE = 'twitter.json'
GET_NEW_DATA = p["get_new_data"]

# model training related stuff
TRAIN = p["train"]
EPOCHS = p["epochs"]
EMBEDDING_DIM = p["embedding_dim"]
UNITS = p["units"]
SEQ_LENGTH = p["seq_length"]
BATCH_SIZE = p["batch_size"]
BUFFER_SIZE = p["buffer_size"]

# output generation related stuff
GENERATE = p["generate"]
NUM_GENERATE = p["num_generate"]
TEMPERATURE = p["temperature"]
################################################################################

# 1. load or download data
input_file = TWITTER_SCREEN_NAME+'.txt'
input_path = 'data/twitter/'
if os.path.isfile(input_path+input_file) and not GET_NEW_DATA:

    print('\nLoading data...')
    with open(input_path+input_file, "rt") as fin:
        input_corpus = fin.read()
    print('done!')

else:

    # create data folder in case it does not exist
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # load credential files
    with open('credentials/'+CREDENTIALS_FILE,"rt") as fin:
        credentials = json.load(fin)

    # download new tweets
    print('\nDownloading tweets...')
    tweets = get_tweets_from_screen_name(TWITTER_SCREEN_NAME, credentials)
    print('done!')

    # add some preprocessing
    input_corpus = tweets_preprocessing(tweets)

    # save new tweets to disc
    with open(input_path+input_file, "wt") as fout:
        fout.write(input_corpus)

vocab = sorted(set(input_corpus))
vocab_size = len(vocab)
print('Total number of characters: {}'.format(len(input_corpus)))
print('Number of different characters: {}'.format(len(vocab)))

##########################################################
# 2. create input for the model and auxiliary dictionaries

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in input_corpus])

# Create training examples / targets
if TRAIN:

    chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(SEQ_LENGTH+1, drop_remainder=True)
    dataset = chunks.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # create model
    model = Model(vocab_size, EMBEDDING_DIM, UNITS)
    model.build(tf.TensorShape([BATCH_SIZE, SEQ_LENGTH]))
    print('\nCreate model summary:')
    model.summary()
    optimizer = tf.train.AdamOptimizer() # define optimizer
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

# Directory where the checkpoints will be saved
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_dir = './checkpoints/tweets_'+TWITTER_SCREEN_NAME
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

###########################################
# 3. train the model or load a previous one
if TRAIN:
    print('\n\nTraining the model...\n\n')

    for epoch in range(EPOCHS):

        start = time.time()

        # initializing the hidden state at the start of every epoch
        hidden = model.reset_states()

        for (batch, (inp, target)) in enumerate(dataset):

            with tf.GradientTape() as tape:
                 # feeding the hidden state back into the model
                 predictions = model(inp)
                 loss = loss_function(target, predictions)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            if batch % 50 == 0:
               print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                                batch,
                                                                loss))
        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        print ('Time taken for 1 epoch {} sec\n'.format(int(time.time() - start)))

    checkpoint.save(file_prefix = checkpoint_prefix)

elif GENERATE:

    print('\n\nLoading previous model...')

# the model always have to lead the weigths because the input dimention changes
# from BATCH_SIZE (when training) to 1 (when generating)
model = Model(vocab_size, EMBEDDING_DIM, UNITS)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
print('done!')

########################
# 4. generate new tweets
if GENERATE:
    print('\n\nGenerating new tweets...')
    start_string = ' ' # character that starts the output
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()
    for i in range(NUM_GENERATE):

        predictions = model(input_eval)

        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / TEMPERATURE
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

        # pass the predicted char as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    output_corpus = start_string + ''.join(text_generated)
    print('done!')

###############################
# 5. save output tweets to disc
if GENERATE:
    print('\n\nSample output: {}'.format(output_corpus[:1000]))
    output_path = 'sample_outputs/twitter_'
    if not os.path.exists('sample_outputs'):
        os.makedirs('sample_outputs')
    with open(output_path+TWITTER_SCREEN_NAME+'_t{}.txt'.format(TEMPERATURE), "wt") as fout:
        fout.write(output_corpus)
