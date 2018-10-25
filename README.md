# Generating tweets with recurrent networks

This repository is my first test for generating text (and tweets) with deep recurrent networks. Most of the code here is based on [this tensorflow tutorial](https://www.tensorflow.org/tutorials/sequences/text_generation) and the classic article [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

The repository is currently under construction, so don't be surprised if part of the documentation is missing or the code is poorly written. Updates are being made as time allows, and suggestions are welcome :).

## Project

This is a (rather hastly put together) repository for training a recurrent neural network with the latest tweets from some user. The recurrent network is used as a generative model, 'creating' new tweets based on the previous learned ones. This models work at the character level, i.e., it predicts what is the next character in the tweet based on the previous ones.

## Data

The data is extracted from twitter using their own [API](https://developer.twitter.com/content/developer-twitter/en.html). To access data from twitter, you need to register as a developer and create an app in their plataform. You will then receive the necessary credentials to access their database, which are required for training this model. The credentials should be included in the `/credentials/twitter.json` file, which is read by the `get_tweets_from_screen_name` function in the `utils.py` script.

Some preprocessing is required before feeding the data to the network, but I have tried to keep it minimum. For now, only html links are completely removed from the dataset, and end of sentence chars are added at the end of each tweet for a clerer output.

## Installation

### Installing git and forking the repository

Make sure you have installed git. Fork a copy of this repository to your own GitHub account and clone your fork into your computer, inside your favorite folder.

### Setting up the environemnt

Install [Python 3.7](https://www.python.org/) and the [conda package manager](https://conda.io/miniconda.html). You set up an envirment to avoid conflicts between different versions of packages. To do so, navigate to the project directory inside a terminal and create a virtual environment (replace <environment_name>, for example, with "environment_name") and install the required packages:

`conda env create -f environment.yml`

Activate the virtual environment:

`source activate twitter_env`

By installing these packages in a virtual environment, you avoid dependency clashes with other packages that may already be installed elsewhere on your computer.

### Using the model

To run the model, simply navigate to the main folder and type `python3 twitter_username.py`. This script loads parameters, network model, and auxiliary functions. The important parameters, which can be changed in `params.json`, are:

Data related parameters:

* `screen_name`: str, the twitter screen name of the twitter user (in the case for the Brazilian elections, choose either "Haddad_Fernando" or "jairbolsonaro")

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

Regarding time, simulating the model for 50 epochs in a reasonably fast laptop (i7-7500U, but no GPU) takes approximately 5 hours. Be mindful of this fact when running the model.

## Sample results

To test the model, I have used as examples the two candidates for the 2nd round of the 2018 Brazilian presidential election. Namely, the candidates are [Fernando Haddad](https://twitter.com/Haddad_Fernando) and [Jair Bolsonaro](https://twitter.com/jairbolsonaro). Of course, the model can be applied to any particular twitter user of your preference, as long as you have access to their data.

The default parameters already result in some reasonably fun output tweets, and also show a bit of what such 'simple' model can learn. The outputs contain a few mistakes and typos (for a temperature of 0.7), of course, as the model has to learn the whole Portuguese language based only on a set of approx. 3000 tweets. Some tweets are grammatically non-sense, and most tweets lack any meaning at all.

First, let's start with the tweets from Haddad (chosen by alphabetical order, and not by any political affiliation, inspiration, admiration, aspiration, or computation). Nice looking output tweets are (outputs only in Portuguese, as this is the language the candidates typically tweet):

`Eu vivo de salário, sou um defensor árduo da democracia. Repudio qualquer forma de autoritarismo, apenas para o segundo turno, para lutarmos juntos em defesa da democracia e da liberdade!`

`Nós queremos apoiar o ensino médio. A juventude que pensa e debater o país. Com assistência médica, enfermaria se for preciso para debater o Brasil. Ninguém pode ser eleito sem apresentar as suas propostas para bicicleta em meu Plano de Governo`

`Agora o povo não consegue empreender... Por isso pro @LulaOficial foi o maior presidente do Brasil. #MaisLivrosMenosArmas`

Now let's look at what nice tweets from Bolsonaro tell us:

`Jair Bolsonaro recebe visita do PSDB com o PT contra Jair Bolsonaro.`

`- Obrigado Santo Amaro da Impeachment, povo nas ruas, Jair Bolsonaro: "Especialistas e eduais o silêncio das campanhas de marketing contra a verdade também nesta guerra pelas várias face de um jornalista investigativo feriu e a casa caia. Essa é a direita é o melhor caminho! É o Bolsonaro?`

`Jair Bolsonaro recebe visita do PSDB contra o combo da piada está visitá-lo imprensa. Quando adotamos o espírito na lista de Janot, Poder`

For those of you who speak portuguese, the tweets must look at least a bit interesting, if not funny. For those who don't, you will miss many references to the current Brazilian popular culture. Maybe you are even able to find some typical traits of the candidates. Interestingly, this 'simple' model is already able to recognize that tweets should be short, contain punctuation, and begin sentences with capital letters. A bit more impressive, typical abbreviations are also present (such as PSDB, PT, and other Brazilian parties). Some tweets contain hashtags, others contain mentions to other users). Shortcommings of the current version of the model are some long range correlations: for example, the model not always closes the quotations it opens, and some subjects do not agree with their verbs in form and gender. Many of these problems can hopefully be removed by adding more layers to the model, or trying different training techniques. This results section will be updated when these new features are added to this project.

Any political implications of such outputs are left to the reader :)

## License

This project is licensed under the MIT License - see LICENSE.md for details.
