import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
from os import getcwd

nltk.download('twitter_samples')
nltk.download('stopwords')

def test_sigmoid(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {"name": "default_check", "input": {"z": 0}, "expected": 0.5},
        {
            "name": "positive_check",
            "input": {"z": 4.92},
            "expected": 0.9927537604041685,
        },
        {"name": "negative_check", "input": {"z": -1}, "expected": 0.2689414213699951},
        {
            "name": "larger_neg_check",
            "input": {"z": -20},
            "expected": 2.0611536181902037e-09,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output from sigmoid function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_gradientDescent(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "random_seed": 1,
                "input_dict": {
                    "x": np.array(
                        [
                            [1.00000000e00, 8.34044009e02, 1.44064899e03],
                            [1.00000000e00, 2.28749635e-01, 6.04665145e02],
                            [1.00000000e00, 2.93511782e02, 1.84677190e02],
                            [1.00000000e00, 3.72520423e02, 6.91121454e02],
                            [1.00000000e00, 7.93534948e02, 1.07763347e03],
                            [1.00000000e00, 8.38389029e02, 1.37043900e03],
                            [1.00000000e00, 4.08904499e02, 1.75623487e03],
                            [1.00000000e00, 5.47751864e01, 1.34093502e03],
                            [1.00000000e00, 8.34609605e02, 1.11737966e03],
                            [1.00000000e00, 2.80773877e02, 3.96202978e02],
                        ]
                    ),  
                    "y": np.array(
                        [
                            [1.0],
                            [1.0],
                            [0.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [1.0],
                        ]
                    ),  
                    "theta": np.zeros((3, 1)),
                    "alpha": 1e-8,
                    "num_iters": 700,
                },
            },
            "expected": {
                "J": 0.6709497038162118,
                "theta": np.array(
                    [[4.10713435e-07], [3.56584699e-04], [7.30888526e-05]]
                ),
            },
        },
        {
            "name": "larger_check",
            "input": {
                "random_seed": 2,
                "input_dict": {
                    "x": np.array(
                        [
                            [1.0, 435.99490214, 25.92623183, 549.66247788],
                            [1.0, 435.32239262, 420.36780209, 330.334821],
                            [1.0, 204.64863404, 619.27096635, 299.65467367],
                            [1.0, 266.8272751, 621.13383277, 529.14209428],
                            [1.0, 134.57994534, 513.57812127, 184.43986565],
                            [1.0, 785.33514782, 853.97529264, 494.23683738],
                            [1.0, 846.56148536, 79.64547701, 505.24609012],
                            [1.0, 65.28650439, 428.1223276, 96.53091566],
                            [1.0, 127.1599717, 596.74530898, 226.0120006],
                            [1.0, 106.94568431, 220.30620707, 349.826285],
                            [1.0, 467.78748458, 201.74322626, 640.40672521],
                            [1.0, 483.06983555, 505.23672002, 386.89265112],
                            [1.0, 793.63745444, 580.00417888, 162.2985985],
                            [1.0, 700.75234661, 964.55108009, 500.00836117],
                            [1.0, 889.52006395, 341.61365267, 567.14412763],
                            [1.0, 427.5459633, 436.74726303, 776.559185],
                            [1.0, 535.6041735, 953.74222694, 544.20816015],
                            [1.0, 82.09492228, 366.34240168, 850.850504],
                            [1.0, 406.27504305, 27.20236589, 247.177239],
                            [1.0, 67.14437074, 993.85201142, 970.58031338],
                        ]
                    ),  
                    "y": np.array(
                        [
                            [1.0],
                            [1.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [1.0],
                            [0.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [1.0],
                            [1.0],
                            [0.0],
                            [0.0],
                            [1.0],
                            [0.0],
                        ]
                    ),  
                    "theta": np.zeros((4, 1)),
                    "alpha": 1e-4,
                    "num_iters": 30,
                },
            },
            "expected": {
                "J": 6.5044107216556135,
                "theta": np.array(
                    [
                        [9.45211976e-05],
                        [2.40577958e-02],
                        [-1.77876847e-02],
                        [1.35674845e-02],
                    ]
                ),
            },
        },
    ]

    for test_case in test_cases:
        # Setting the random seed for reproducibility
        result_J, result_theta = target(**test_case["input"]["input_dict"])

        try:
            assert isinstance(result_J, float)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": float, "got": type(result_J),}
            )
            print(
                f"Wrong output type for loss function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result_J, test_case["expected"]["J"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["J"],
                    "got": result_J,
                }
            )
            print(
                f"Wrong output for the loss function. Check how you are implementing the matrix multiplications. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_theta.shape == test_case["input"]["input_dict"]["theta"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["input"]["input_dict"]["theta"].shape,
                    "got": result_theta.shape,
                }
            )
            print(
                f"Wrong shape for weights matrix theta. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                np.squeeze(result_theta), np.squeeze(test_case["expected"]["theta"]),
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["theta"],
                    "got": result_theta,
                }
            )
            print(
                f"Wrong values for weight's matrix theta. Check how you are updating the matrix of weights. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
    


# +
def test_extract_features(target, freqs):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "tweet": "#FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)",
                "freqs": freqs,
            },
            "expected": np.array(
                [[1.00e00, 3.133e03, 6.10e01]]
            ), 
        },
        {
            "name": "unk_words_check",
            "input": {"tweet": "blorb bleeeeb bloooob", "freqs": freqs},
            "expected": np.array([[1.0, 0.0, 0.0]]),
        },
        {
            "name": "good_words_check",
            "input": {"tweet": "Hello world! All's good!", "freqs": freqs},
            "expected": np.array([[1.0, 263.0, 106.0]]),
        },
        {
            "name": "bad_words_check",
            "input": {"tweet": "It is so sad!", "freqs": freqs},
            "expected": np.array([[1.0, 5.0, 100.0]]),
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result.shape == test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Wrong output shape. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output values. Check how you are computing the positive or negative word count. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


# -

def test_predict_tweet(target, freqs, theta):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check1",
            "input": {"tweet": "I am happy", "freqs": freqs, "theta": theta},
            "expected": np.array([[0.5192746]]),
        },
        {
            "name": "default_check2",
            "input": {"tweet": "I am bad", "freqs": freqs, "theta": theta},
            "expected": np.array([[0.49434685]]),
        },
        {
            "name": "default_check3",
            "input": {
                "tweet": "this movie should have been great",
                "freqs": freqs,
                "theta": theta,
            },
            "expected": np.array([[0.5159792]]), 
        },
        {
            "name": "default_check5",
            "input": {"tweet": "It is a good day", "freqs": freqs, "theta": theta,},
            "expected": np.array([[0.52320595]]), 
        },
        {
            "name": "default_check6",
            "input": {"tweet": "It is a bad bad day", "freqs": freqs, "theta": theta,},
            "expected": np.array([[0.49780224]]), 
        },
        {
            "name": "default_check7",
            "input": {
                "tweet": "It is a good day",
                "freqs": freqs,
                "theta": np.array([[5.0000e-04], [-3.4e-02], [3.2e-02]]),
            },
            "expected": np.array([[0.00147813]]), 
        },
        {
            "name": "default_check8",
            "input": {
                "tweet": "It is a bad bad day",
                "freqs": freqs,
                "theta": np.array([[5.0000e-04], [-3.4e-02], [3.2e-02]]),
            },
            "expected": np.array([[0.45673348]]), 
        },
        {
            "name": "default_check9",
            "input": {
                "tweet": "this movie should have been great",
                "freqs": freqs,
                "theta": np.array([[5.0000e-04], [-3.4e-02], [3.2e-02]]),
            },
            "expected": np.array([[0.01561938]]), 
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result.shape == test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Wrong output shape. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong predicted values. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
    


def unittest_test_logistic_regression(target, freqs, theta):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check1",
            "input": {
                "test_x": [
                    "Bro:U wan cut hair anot,ur hair long Liao bo\nMe:since ord liao,take it easy lor treat as save $ leave it longer :)\nBro:LOL Sibei xialan",
                    "@heyclaireee is back! thnx God!!! i'm so happy :)",
                    "@BBCRadio3 thought it was my ears which were malfunctioning, thank goodness you cleared that one up with an apology :-)",
                    "@HumayAG 'Stuck in the centre right with you. Clowns to the right, jokers to the left...' :) @orgasticpotency @ahmedshaheed @AhmedSaeedGahaa",
                    "Happy Friday :-) http://t.co/iymPIlWXFY",
                    "I wanna change my avi but uSanele :(",
                    "MY PUPPY BROKE HER FOOT :(",
                    "where's all the jaebum baby pictures :((",
                    "But but Mr Ahmad Maslan cooks too :( https://t.co/ArCiD31Zv6",
                    "@eawoman As a Hull supporter I am expecting a misserable few weeks :-(",
                ],
                "test_y": np.array(
                    [
                        [1.0],
                        [1.0],
                        [1.0],
                        [1.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ]
                ),
                "freqs": freqs,
                "theta": theta,
            },
            "expected": 1.0,
        },
        {
            "name": "default_check1",
            "input": {
                "test_x": [
                    "Bro:U wan cut hair anot,ur hair long Liao bo\nMe:since ord liao,take it easy lor treat as save $ leave it longer :)\nBro:LOL Sibei xialan",
                    "@heyclaireee is back! thnx God!!! i'm so happy :)",
                    "@BBCRadio3 thought it was my ears which were malfunctioning, thank goodness you cleared that one up with an apology :-)",
                    "@HumayAG 'Stuck in the centre right with you. Clowns to the right, jokers to the left...' :) @orgasticpotency @ahmedshaheed @AhmedSaeedGahaa",
                    "Happy Friday :-) http://t.co/iymPIlWXFY",
                    "I wanna change my avi but uSanele :(",
                    "MY PUPPY BROKE HER FOOT :(",
                    "where's all the jaebum baby pictures :((",
                    "But but Mr Ahmad Maslan cooks too :( https://t.co/ArCiD31Zv6",
                    "@eawoman As a Hull supporter I am expecting a misserable few weeks :-(",
                ],
                "test_y": np.array(
                    [
                        [1.0],
                        [1.0],
                        [1.0],
                        [1.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ]
                ),
                "freqs": freqs,
                "theta": np.array([[5.0000e-04], [-3.4e-02], [3.2e-02]]),
            },
            "expected": 0.0,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, np.float64)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.float64,
                    "got": type(result),
                }
            )
            print(
                f"Wrong output type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong accuracy value. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

  def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks    
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set) 
test_pos = positive_tweets[4000:]
train_pos = positive_tweets[:4000]
test_neg = negative_tweets[4000:]
train_neg = negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# Print the shape train and test sets
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

# test the function below
print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))

# Implementing the sigmoid function.
def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))  
    return h

# Testing your function 
if (sigmoid(0) == 0.5):
    print('SUCCESS!')
else:
    print('Oops!')

if (sigmoid(4.92) == 0.9927537604041685):
    print('CORRECT!')
else:
    print('Oops again!')

# Test your function
test_sigmoid(sigmoid)

# verify that when the model predicts close to 1, but the actual label is 0, the loss is a large positive value
-1 * (1 - 0) * np.log(1 - 0.9999) 
# verify that when the model predicts close to 0 but the actual label is 1, the loss is a large positive value
-1 * np.log(0.0001)

# Implementing gradient descent function
def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    # get 'm', the number of rows in matrix x
    m = x.shape[0]     
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x,theta)
        
        # get the sigmoid of h
        h = sigmoid(z)
        
        # calculate the cost function
        J = -1./m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(),np.log(1-h)))                                                    

        # update the weights theta
        theta = theta - (alpha/m) * np.dot(x.transpose(),(h-y))

    J = float(J)
    return J, theta

# Check the function
# Construct a synthetic test case using numpy PRNG functions
np.random.seed(1)
# X input is 10 x 3 with ones for the bias terms
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
# Y Labels are 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

# Apply gradient descent
tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")

test_gradientDescent(gradientDescent)

# Extracting the Features
def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
    
    # loop through each word in the list of words
    # loop through each word in the list of words
    for word in word_l:
        
        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word, 1.0),0)
        
        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word, 0.0),0)
        
    assert(x.shape == (1, 3))
    return x

# Check your function
# test 1
# test on training data
tmp1 = extract_features(train_x[0], freqs)
print(tmp1)

# test 2:
# check for when the words are not in the freqs dictionary
tmp2 = extract_features('blorb bleeeeb bloooob', freqs)
print(tmp2)

test_extract_features(extract_features, freqs)

# Training Model
# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

# Testing Model
def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    
    # extract the features of the tweet and store it into x
    x = extract_features(tweet,freqs)
    
    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x,theta))
    
    return y_pred

# Run this cell to test your function
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))   

# Feel free to check the sentiment of your own tweet below
my_tweet = 'I am learning :)'
predict_tweet(my_tweet, freqs, theta)

# Feel free to check the sentiment of your own tweet below
my_tweet = 'Yor are such a crap :)'
predict_tweet(my_tweet, freqs, theta)

# Checking performance using Test Set
def test_logistic_regression(test_x, test_y, freqs, theta, predict_tweet=predict_tweet):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    # the list for storing predictions
    y_hat = []
    
    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)
        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1)
        else:
            # append 0 to the list
            y_hat.append(0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)
    
    return accuracy

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")
unittest_test_logistic_regression(test_logistic_regression, freqs, theta)

# Error Analysis
print('Label Predicted Tweet')
for x,y in zip(test_x,test_y):
    y_hat = predict_tweet(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_tweet(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))

  my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')

my_tweet = 'You are such a jerk!'
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')
