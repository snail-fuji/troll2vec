# Toxic comments detection

*Based on Alexander Rashkin keras implementation of Yoon Kim CNN for natural texts classification*

*Prepared for Social Weekend Hackaton 13*

Related to [the extension](https://github.com/lexaich/SNAT)

# Deployment and usage

First, download dataset, preprocess it and train a model.

```bash
$ kaggle competitions download jigsaw-toxic-comment-classification-challenge
$ unzip train.csv.zip
$ jupyter nbconvert --execute ./dataset.ipynb
$ python3 ./train.py
```
After these steps you will be able to run flask server

```bash
$ python3 ./server.py
```

Example of server response goes below:
```bash
$ curl -XPOST -d '{"messages": ["Your mom is so fat", "Your mom is so cool"]}' localhost:5000/api
{
  "toxicity": [
    true, 
    false
  ]
}
```

# Convolutional neural network

Train convolutional network for toxicity detection. Based on "Convolutional Neural Networks for Sentence Classification" by Yoon Kim, [link](http://arxiv.org/pdf/1408.5882v2.pdf). Inspired by Denny Britz article "Implementing a CNN for Text Classification in TensorFlow", [link](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).
For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85%

## Some difference from original article:
* larger corpus, longer sentences; sentence length is very important, just like data size
* smaller embedding dimension, 20 instead of 300
* 2 filter sizes instead of original 3
* much fewer filters; experiments show that 3-10 is enough; original work uses 100
* random initialization is no worse than word2vec init on IMDB corpus
* sliding Max Pooling instead of original Global Pooling

## Dependencies

* The [Keras](http://keras.io/) Deep Learning library and most recent [Theano](http://deeplearning.net/software/theano/install.html#install) backend should be installed. You can use pip for that. 
Not tested with TensorFlow, but should work.
