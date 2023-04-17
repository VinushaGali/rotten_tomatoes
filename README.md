1) Download the Dataset from Kaggle
2) Upload it into the Google Drive.
3) Load the Python Program into the Google Colab and run it.

PROGRAM IMPLEMENTATION
1) Import necessary libraries
2) Define custom list of stop words
3) Read the dataset
4) Split the data into train, dev, and test sets: The dataset is split into train and test sets using train_test_split function with a 0.2 test size and a random state of 42. Then, the test set is further split into dev and test sets with a 0.5 test size and a random state of 42.
5) Build a vocabulary: The program builds a vocabulary from the words in the training set that occur at least five times and are not in the custom list of stop words. It uses a defaultdict to count the occurrences of each word in the training set, and then creates the vocabulary by selecting the words that meet the criteria.
6) Calculate the probabilities: It then calculates the probabilities of each word appearing in a positive or negative review using a smoothing parameter, alpha by using Laplace Smoothing. It uses a defaultdict to count the occurrences of each word in positive and negative reviews separately, and then calculates the probabilities for each word. 
7) Calculate accuracy: The program calculates the accuracy of the predictions by comparing the predicted labels with the actual labels in the test set.
8) Print results: The program prints out the accuracy and the top 10 words that predict each class for each alpha value.
