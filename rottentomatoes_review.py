import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')

# Define a custom list of stop words based on the domain of the dataset
my_stop_words = set(['film', 'movie', 'actor', 'director', 'the', 'for', 'is', 'like', 'and', 'a', 'of', 'to', 'is', 'in', 'that', 'with', 'it', 'as', 'but', 'its'])

# read the dataset
df = pd.read_csv("/content/drive/MyDrive/rt_reviews.csv")

# split the data into train, dev, and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)
dev, test = train_test_split(test, test_size=0.5, random_state=42)

# build a vocabulary
# I referred https://www.accelebrate.com/blog/using-defaultdict-python for defaultdict()
word_counts = defaultdict(int)
for review in train["Review"]:
    for word in review.split():
        if word.lower() not in my_stop_words:  # exclude custom stop words from the vocabulary
            word_counts[word.lower()] += 1

vocab = [word for word in word_counts if word_counts[word] >= 5]
word_index = {word: i for i, word in enumerate(vocab)}

# calculate the probabilities
num_documents = len(train)
num_positives = len(train[train["Freshness"] == "fresh"])
num_negatives = len(train[train["Freshness"] == "rotten"])

# I referred https://gamedevacademy.org/text-classification-tutorial-with-naive-bayes/ for the below code

alpha_values = [0.1, 1.0, 10.0]
for alpha in alpha_values:
    print("Alpha value:", alpha)
    
    word_count_pos = defaultdict(int)
    word_count_neg = defaultdict(int)
    for index, row in train.iterrows():
        for word in row["Review"].split():
            if word.lower() in vocab:
                if row["Freshness"] == "fresh":
                    word_count_pos[word.lower()] += 1
                else:
                    word_count_neg[word.lower()] += 1

    prob_word_pos = {word: (word_count_pos[word] + alpha) / (num_positives + alpha*len(vocab)) for word in vocab}
    prob_word_neg = {word: (word_count_neg[word] + alpha) / (num_negatives + alpha*len(vocab)) for word in vocab}

    # predict the test set labels
    
    predictions = []
    for index, row in test.iterrows():
        prob_pos_review = num_positives / num_documents
        prob_neg_review = num_negatives / num_documents
        for word in row["Review"].split():
            if word.lower() in vocab:
                prob_pos_review *= prob_word_pos[word.lower()]
                prob_neg_review *= prob_word_neg[word.lower()]

        if prob_pos_review > prob_neg_review:
            predictions.append("fresh")
        else:
            predictions.append("rotten")

    # calculate accuracy
    correct = 0
    total = len(test)
    for i in range(total):
        if predictions[i] == test.iloc[i]["Freshness"]:
            correct += 1

    accuracy = correct / total
    print("Accuracy:", accuracy)

    # top 10 words that predict each class
    top_words_pos = [word for word in sorted(prob_word_pos, key=prob_word_pos.get, reverse=True) if word.lower() not in my_stop_words][:10]
    top_words_neg = [word for word in sorted(prob_word_neg, key=prob_word_neg.get, reverse=True) if word.lower() not in my_stop_words][:10]

    print("Top 10 words that predict 'fresh':", top_words_pos)
    print("Top 10 words that predict 'rotten':", top_words_neg) 
    print()
