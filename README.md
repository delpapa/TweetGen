# Generating tweets with recurrent networks

This repository is my first test for generating text (and tweets) with deep recurrent networks. Most of the code here is based on [this tensorflow tutorial](https://www.tensorflow.org/tutorials/sequences/text_generation) and the classic article [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

The repository is currently under construction, so don't be surprised if part of the documentation is missing or the code is poorly written. Updates are being made as time allows, and suggestions are welcome :).

## Project

This is a (rather hastly put together) repository for training a recurrent neural network with the latest tweets from the two candidates for the 2018 Brazilian presidential election (2nd round). Namely, the candidates are [Fernando Haddad](https://twitter.com/Haddad_Fernando) and [Jair Bolsonaro](https://twitter.com/jairbolsonaro). Of course, the model can be applied to any particular twitter user of your preference, as long as you have access to their data. 

The recurrent network is used as a generative model, 'creating' new tweets based on the previous learned ones. This models work at the character level, i.e., it predicts what is the next character in the tweet based on the previous ones.

## Data

The data is extracted from twitter using their own [API](https://developer.twitter.com/content/developer-twitter/en.html). To access data from twitter, you need to register as a developer and create an app in their plataform. You will then receive the necessary credentials to access their database, which are required for training this model. The credentials should be included in the `/credentials/twitter.json` file, which is read by the `get_tweets_from_screen_name` function in the `utils.py` script.

## Installation 

### Installing git and forking the repository

Make sure you have installed git. Fork a copy of this repository to your own GitHub account and clone your fork into your computer, inside your favorite folder.

### Setting up the environemnt

Install [Python 3.6](https://www.python.org/) and the [conda package manager](https://conda.io/miniconda.html). You set up an envirment to avoid conflicts between different versions of packages. To do so, navigate to the project directory inside a terminal and create a virtual environment (replace <environment_name>, for example, with "environment_name") and install the required packages:

`conda create -n <environment_name> --file requirements.txt python=3.6`

Activate the virtual environment:

`source activate <environment_name>`

By installing these packages in a virtual environment, you avoid dependency clashes with other packages that may already be installed elsewhere on your computer.

### Using the model

To run the model, simply navigate to the main folder and type `python3 twitter_username.py`. This script loads parameters, network model, and auxiliary functions. The important parameters, which can be changed in `params.json`, are:

Data related parameters:

* `screen_name`: str, the twitter screen name of the twitter user (in this case, choose either "Haddad_Fernando" or "jairbolsonaro")

* `get_new_data`: bool, if True, get the newest ~ 3000 tweets from the user

Training related parameters:
* `train`: bool, if True, train the model; if False, load the latest previously trained model
* `epochs`: int, number of epochs to train the model 
* `embedding_dim`: int, number of embedding dimensions for each character
* `units`: int, number of units for the recurrent layer
* `seq_length`: int, maximum sequence length for a single input in characters
* `batch_size`: int, the batch size
* `buffer_size`: int, size of the buffer when shuffling the input dataset (necessary due to tensorflow implementation)

Generation related parameters:
* `generate`: bool, if True, generate a new output with the latest previously trained model
* `temperature`: float, between 0 and 1: how predictable the output text it (1 for more surprising outputs)

Please note that the default values (as well as the network architecture) are inspired by the [tensorflow tutorial](https://www.tensorflow.org/tutorials/sequences/text_generation), and by no means are guaranteed to return the best results. Model tuning is still ongoing work! Feel free to play with these parameters and maybe let me know what works best for you.

## Sample results

The default parameters already result in some reasonably fun output tweets, and also show a bit of what such 'simple' model can learn. First, let's start with the tweets from Haddad (chosen by alphabetical order, and not by any political affiliation, inspiration, admiration, aspiration, or computation). Nice looking output tweets are:



Now let's look at that the tweets from Bolsonaro have to say:



The political implications of such outputs are left to the reader :)

## License

This project is licensed under the MIT License - see LICENSE.md for details.
