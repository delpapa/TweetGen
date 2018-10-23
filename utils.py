import tweepy
import re


def get_tweets_from_screen_name(screen_name, credentials):
    """
    Get the last 3240 tweets (maximum allowed by the API) from an user
    with given screen name.
    Adapted from https://gist.github.com/yanofsky/5436496

    Parameters:
        screen_name: str, the screen_name of the user (ex: @random_user becomes
                          'random_user')
        credentials: dic, contain the credentials for accessing twitter. Must
                          include the keys 'consumer_API_key',
                          'consumer_API_secret_key', 'access_token', and
                          'access_secret_token'
    Returns:
        tweets: list, list of Status objects containing the last 3240 of given user
    """

    # API authentication
    auth = tweepy.OAuthHandler(credentials['consumer_API_key'],
                               credentials['consumer_API_secret_key'])
    auth.set_access_token(credentials['access_token'],
                          credentials['access_secret_token'])
    api = tweepy.API(auth)


    tweets = []
    while True:

        try:
            new_tweet = api.user_timeline(screen_name=screen_name,
                                          count=200,
                                          max_id=oldest,
                                          tweet_mode="extended")
        except:
            #all subsiquent requests use the max_id param to prevent duplicates
            new_tweet = api.user_timeline(screen_name=screen_name,
                                          count=200,
                                          tweet_mode="extended")
        if len(new_tweet) == 0:
            break

        tweets.extend(new_tweet)
        oldest = tweets[-1].id - 1  #update the id of the oldest tweet less one

    return tweets


def tweets_preprocessing(tweets):

    tweet_corpus = []
    for tweet in tweets:

        # include retweets
        if hasattr(tweet, 'retweeted_status'):
            tweet_corpus.extend(tweet.retweeted_status.full_text+u"\u0004 "
)
        else:
            tweet_corpus.extend(tweet.full_text+u"\u0004 "
)

    tweet_corpus = ''.join(tweet_corpus)
    # remove links
    # from https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    tweet_corpus = re.sub(r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*',
                          '', tweet_corpus, flags=re.MULTILINE)

    return tweet_corpus
