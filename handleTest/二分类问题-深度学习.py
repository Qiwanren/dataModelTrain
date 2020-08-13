from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=8000) #前8000个单词(每局评论至多包含8000个单词)