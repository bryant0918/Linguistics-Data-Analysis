import numpy as np
import pandas as pd
import regex as re
import io
import shutil
import csv
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import itertools
import spacy
from IPython.display import display


class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''

    def __init__(self):
        self.data = None
        self.ham_count = None
        self.spam_count = None
        self.words = None

        return

    def tokenize(self, message, words=None):
        if words is None:
            return message.split()
        else:
            return [w if w in words else "<UNK>" for w in message.split()]

    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # Get dictionary of words
        words_label = [(self.tokenize(X.iloc[i]), y.iloc[i]) for i in range(len(X))]
        self.words = set(itertools.chain.from_iterable([tup[0] for tup in words_label]))
        word_dict = {word:{"spam": 0, "ham": 0} for word in self.words}
        word_dict['<UNK>'] = {"spam": 1, "ham": 1}

        # Start counting
        self.ham_count = 0
        self.spam_count = 0
        for words, label in words_label:
            if label == "spam":
                self.spam_count += 1
            elif label == "ham":
                self.ham_count += 1
            for word in words:
                word_dict[word][label] += 1

        # Create dataframe
        self.data = pd.DataFrame(word_dict)


    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        # Find the probabilities
        prob_spam = self.spam_count / (self.spam_count + self.ham_count)
        prob_ham = self.ham_count / (self.ham_count + self.spam_count)

        # Multiply by conditional probabilities
        spam_probs = [prob_spam*np.product([self.data.loc['spam'][xi]/self.data.loc['spam'].sum(axis=0) for xi in self.tokenize(X.iloc[i], self.words)]) for i in range(len(X))]
        ham_probs = [prob_ham*np.product([self.data.loc['ham'][xi]/self.data.loc['ham'].sum(axis=0) for xi in self.tokenize(X.iloc[i], self.words)]) for i in range(len(X))]

        return np.array(np.column_stack((ham_probs, spam_probs)))


    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Get probabilities and take max
        probabilities = self.predict_proba(X)
        binary_labels = np.array([np.argmax(probabilities[i]) for i in range(len(probabilities))])

        return np.array(['ham' if idx == 0 else "spam" for idx in binary_labels])

    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''

        # Find the probabilities
        prob_spam = np.log(self.spam_count / (self.spam_count + self.ham_count))
        prob_ham = np.log(self.ham_count / (self.ham_count + self.spam_count))

        # Add to conditional probabilities
        spam_probs = [prob_spam + np.sum([np.log((self.data.loc['spam'][xi] + 1) / (self.data.loc['spam'].sum(axis=0) + 2)) for xi in
                                              self.tokenize(X.iloc[i], self.words)]) for i in range(len(X))]
        ham_probs = [prob_ham + np.sum([np.log((self.data.loc['ham'][xi] + 1) / (self.data.loc['ham'].sum(axis=0) + 2)) for xi in
                                            self.tokenize(X.iloc[i], self.words)]) for i in range(len(X))]

        return np.array(np.column_stack((ham_probs, spam_probs)))

    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Get probabilities and take max
        probabilities = self.predict_log_proba(X)
        binary_labels = np.array([np.argmax(probabilities[i]) for i in range(len(probabilities))])

        return np.array(['ham' if idx == 0 else "spam" for idx in binary_labels])


def sklearn_method(X_train, y_train, X_test, all=False):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''

    # Transform data
    # vectorizer = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1,3), max_features=1000)
    vectorizer = CountVectorizer()

    if all:
        train_counts = vectorizer.fit_transform(X_train["Content"])
        test_counts = vectorizer.transform(X_test["Content"])
    else:
        train_counts = vectorizer.fit_transform(X_train)
        test_counts = vectorizer.fit_transform(X_test)

    # Train model
    clf = MultinomialNB()
    clf = clf.fit(train_counts, y_train)

    # Get prediction labels
    labels = clf.predict(test_counts)

    return labels

def find_features(file_utf8, label='spam'):

    with open(file_utf8, 'r', encoding='utf-8', errors='ignore') as f:
        data = f.read()

        # Convert to bytes to replace some stuff we don't want then decode back to string
        data = data.encode('utf-8').replace(b"\xc2\xa0", b"") # Nonbreaking space
        data = data.replace(b"\xa0", b"") # A weird A symbol
        data = data.decode('utf-8', errors='ignore')

        email_list = re.split(r"From:\t|From: ", data)
        print(len(email_list))

    from_ = re.compile(r'(.*)\n')
    subject_ = re.compile(r'Subject:\s(.*)\n')
    content_ = re.compile(r'\n\n([\s\S]*)')
    reply_to_ = re.compile(r'Reply-To:\s(.*)\n')

    email_dict = []
    for i,email in enumerate(email_list):
        from_feature = re.search(from_, email)
        subject_feature = re.search(subject_, email)
        content_feature = re.search(content_, email[:400])
        reply_to_feature = re.search(reply_to_, email)

        if subject_feature is None:
            subject = "None"
        else:
            subject = subject_feature.groups()[0]

        if content_feature is None:
            content = "None"
        else:
            content = content_feature.groups()[0]

        if reply_to_feature is None:
            reply_to = "None"
        else:
            reply_to = reply_to_feature.groups()[0]

        email_dict.append({"Label": label,
                           "From": from_feature.groups()[0],
                           "Subject": subject,
                           "Reply To": reply_to,
                           "Content": content})

    return email_dict


def append_to_csv(email_dict, header=False):
    csv_columns = ["Label", "From", "Subject", "Reply To", "Content"]
    csv_file = "spam_ham.csv"
    with open(csv_file, 'a', encoding='utf-8', errors='ignore') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns, lineterminator='\n')
        if header:
            writer.writeheader()
        for i,data in enumerate(email_dict):
            writer.writerow(data)

    pass


def create_csv():

    unread_files = ["unread.txt", "unread2.txt", "unread3.txt", "unread4.txt", "unread5.txt", "unread6.txt",
                    "unread7.txt", "unread8.txt", "unread9.txt", "unread10.txt"]
    i = 0
    for file in unread_files:
        email_dict = find_features(file, label='spam')
        if i == 0:
            append_to_csv(email_dict, header=True)
        else:
            append_to_csv(email_dict)
        i += 1
    print("Finished unread")
    read_files = ["opened.txt", "opened2.txt", "opened3.txt", "opened4.txt", "opened5.txt", "opened6.txt"]
    for file in read_files:
        email_dict = find_features(file, label='ham')
        append_to_csv(email_dict)

        # df = pd.DataFrame(email_dict)
        # # df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["", ""], regex=True, inplace=True)
        # df.to_csv("spam_ham.csv")

    pass



def testNB():
    df = pd.read_csv("spam_or_ham.csv")

    # All features
    X = df.loc[:, 'From':'Content']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, shuffle=True)
    labels = sklearn_method(X_train, y_train, X_test, all=True)
    print("Accuracy Score for all features: ", accuracy_score(y_test, labels))

    # All features
    X = df.loc[:, 'From':'Subject Tags_']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, shuffle=True)
    labels = sklearn_method(X_train, y_train, X_test, all=True)
    print("Accuracy Score for ALL features: ", accuracy_score(y_test, labels))


    # From feature
    X = df.loc[:, 'From']
    y = df.loc[:, 'Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, shuffle=True)

    NBF = NaiveBayesFilter()
    NBF.fit(X_train, y_train)
    probs = NBF.predict_proba(X_test)
    labels = NBF.predict_log(X_test)
    print("Accuracy Score My Naive Bayes From: ", accuracy_score(y_test, labels))

    # labels = sklearn_method(X_train, y_train, X_test)
    # print("Accuracy Score for From: ", accuracy_score(y_test, labels))

    # Subject Feature
    X = df['Subject']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, shuffle=True)
    # labels = sklearn_method(X_train, y_train, X_test)
    # print("Accuracy Score for Subject: ", accuracy_score(y_test, labels))
    NBF = NaiveBayesFilter()
    NBF.fit(X_train, y_train)
    probs = NBF.predict_proba(X_test)
    labels = NBF.predict_log(X_test)
    print("Accuracy Score My Naive Bayes Subject: ", accuracy_score(y_test, labels))

    # Subject PoS Feature
    X = df['Subject PoS']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, shuffle=True)
    # labels = sklearn_method(X_train, y_train, X_test)
    # print("Accuracy Score for Subject: ", accuracy_score(y_test, labels))
    NBF = NaiveBayesFilter()
    NBF.fit(X_train, y_train)
    probs = NBF.predict_proba(X_test)
    labels = NBF.predict_log(X_test)
    print("Accuracy Score My Naive Bayes Subject: ", accuracy_score(y_test, labels))


    # Reply To Feature
    X = df['Reply To']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, shuffle=True)
    # labels = sklearn_method(X_train, y_train, X_test)
    # print("Accuracy Score for Subject: ", accuracy_score(y_test, labels))
    NBF = NaiveBayesFilter()
    NBF.fit(X_train, y_train)
    labels = NBF.predict_log(X_test)
    print("Accuracy Score My Naive Bayes Reply To: ", accuracy_score(y_test, labels))

    # Content Feature
    X = df['Content']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, shuffle=True)
    # labels = sklearn_method(X_train, y_train, X_test)
    # print("Accuracy Score for Content: ", accuracy_score(y_NBF = NaiveBayesFilter()
    # NBF.fit(X_train, y_train)
    # labels = NBF.predict_log(X_test)
    # print("Accuracy Score My Naive Bayes Content: ", accuracy_score(y_test, labels))

    pass


def testRF():

    # Transform data
    vectorizer = CountVectorizer()
    df = pd.read_csv("spam_or_ham.csv")


    # Content
    X = df.loc[:, 'Content']
    y = df['Label']

    X_counts = vectorizer.fit_transform(X)
    # stop_words = vectorizer.get_stop_words()
    X_train, X_test, y_train, y_test = train_test_split(X_counts, y, train_size=.7, shuffle=True)

    best_params = [None, None, None]
    best_acc = 0

    for n_estimators in [100, 150, 200, 250]:
        for max_depth in [1, 2, 3, 4, 5]:
            for min_samples_leaf in [1, 2, 3, 4, 5]:
                rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                rfc.fit(X_train, y_train)
                labels = rfc.predict(X_test)
                acc = accuracy_score(y_test, labels)
                if acc > best_acc:
                    best_acc = acc
                    best_params = [n_estimators, max_depth, min_samples_leaf]

    print("Content", best_acc)
    print("Content", best_params)



    # # All
    # X = df.loc[:, 'From':'Content']
    # y = df['Label']
    #
    # X['From'] = vectorizer.fit_transform(X['From'])
    # X['Subject'] = vectorizer.fit_transform(X['Subject'])
    # X['Reply To'] = vectorizer.fit_transform(X['Reply To'])
    # X['Content'] = vectorizer.fit_transform(X['Content'])
    #
    #
    # # stop_words = vectorizer.get_stop_words()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, shuffle=True)
    #
    # best_params = [None, None, None]
    # best_acc = 0
    #
    # for n_estimators in [100, 150, 200, 250]:
    #     for max_depth in [1, 2, 3, 4, 5]:
    #         for min_samples_leaf in [1, 2, 3, 4, 5]:
    #             rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
    #                                          min_samples_leaf=min_samples_leaf)
    #             rfc.fit(X_train, y_train)
    #             labels = rfc.predict(X_test)
    #             acc = accuracy_score(y_test, labels)
    #             if acc > best_acc:
    #                 best_acc = acc
    #                 best_params = [n_estimators, max_depth, min_samples_leaf]
    #
    # print("ALL", best_acc)
    # print("ALL", best_params)

def testVectorizerRF():
    df = pd.read_csv("spam_or_ham.csv")
    # Content
    X = df.loc[:, 'Content']
    y = df['Label']
    best_acc = 0
    for lowercase in [True, False]:
        for stop_words in ['english', None]:
            for ngram_range in [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)]:
                for max_features in [None, 100, 1000, 10000]:
                    vectorizer = CountVectorizer(lowercase=lowercase, stop_words=stop_words, ngram_range=ngram_range,
                                                 max_features=max_features)

                    X_counts = vectorizer.fit_transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(X_counts, y, train_size=.7, shuffle=True)

                    rfc = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=3)
                    rfc.fit(X_train, y_train)
                    labels = rfc.predict(X_test)
                    acc = accuracy_score(y_test, labels)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = [lowercase, stop_words, ngram_range, max_features]

    print("Content", best_acc)
    print("Content", best_params)

def testVectorizerNB():
    df = pd.read_csv("spam_or_ham.csv")
    # Content
    X = df.loc[:, 'Content']
    y = df['Label']
    best_acc = 0
    for lowercase in [True, False]:
        for stop_words in ['english', None]:
            for ngram_range in [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)]:
                for max_features in [None, 100, 1000, 10000]:
                    vectorizer = CountVectorizer(lowercase=lowercase, stop_words=stop_words, ngram_range=ngram_range,
                                                 max_features=max_features)

                    X_counts = vectorizer.fit_transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(X_counts, y, train_size=.7, shuffle=True)

                    # Train model
                    clf = MultinomialNB()
                    clf = clf.fit(X_train, y_train)

                    # Get prediction labels
                    labels = clf.predict(X_test)
                    acc = accuracy_score(y_test, labels)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = [lowercase, stop_words, ngram_range, max_features]

    print("Content", best_acc)
    print("Content", best_params)


def get_pos():
    df = pd.read_csv("spam_ham.csv")
    # Subject
    X = df.loc[:, 'Subject']
    y = df['Label']
    vectorizer = CountVectorizer(lowercase=True, stop_words=None, ngram_range=(1,1), max_features=None)

    X_counts = vectorizer.fit_transform(X)
    features = vectorizer.get_feature_names_out(X)
    print(len(features))

    """
    Change the tag in subject and run on POS
    """
    nlp = spacy.load("en_core_web_lg")

    subject_dict = []
    for subject in X:
        doc = nlp(subject)

        part_of_speechs = np.array([token.pos for token in doc])
        tags = np.array([token.tag for token in doc])

        part_of_speechs_ = np.array([token.pos_ for token in doc])
        tags_ = np.array([token.tag_ for token in doc])

        subject_dict.append({"Subject PoS_": part_of_speechs_,
                             "Subject Tags_": tags_,
                             "Subject PoS": part_of_speechs,
                             "Subject Tags": tags
                             })

        # for token in doc:
        #     print(token.text, token.pos_, token.tag_, token.is_stop)

    subject = pd.DataFrame(subject_dict, columns=["Subject PoS_", "Subject Tags_", "Subject PoS", "Subject Tags"])
    subject.to_csv("testdf.csv", header=True)


    pass

def get_pos2():
    df = pd.read_csv("spam_ham.csv")
    # Subject And Content
    X1 = df.loc[:, 'Subject']
    X2 = df.loc[:, "Content"]

    """
    Change the tag in subject and run on POS
    """
    nlp = spacy.load("en_core_web_lg")

    added_dict = []
    for subject, content in zip(X1,X2):

        subj = nlp(subject)
        cont = nlp(str(content))

        subj_pos = np.array([token.pos_ for token in subj])
        cont_pos = np.array([token.pos_ for token in cont])
        s = len(subj_pos)
        c = len(cont_pos)



        # Adjectives
        num_subj_adjs = np.count_nonzero(subj_pos == "ADJ") if s != 0 else 0
        perc_subj_adjs = num_subj_adjs/s if s != 0 else 0
        num_cont_adjs = np.count_nonzero(cont_pos == "ADJ") if c != 0 else 0
        perc_cont_adjs = num_cont_adjs/c if c != 0 else 0
        num_total_adjs = num_subj_adjs + num_cont_adjs if s+c != 0 else 0
        perc_total_adjs = num_total_adjs/(s+c) if s+c != 0 else 0

        # Adverbs
        num_subj_advs = np.count_nonzero(subj_pos == "ADV") if s != 0 else 0
        perc_subj_advs = num_subj_advs / s if s != 0 else 0
        num_cont_advs = np.count_nonzero(cont_pos == "ADV") if c != 0 else 0
        perc_cont_advs = num_cont_advs / c if c != 0 else 0
        num_total_advs = num_subj_advs + num_cont_advs if s+c != 0 else 0
        perc_total_advs = num_total_advs / (s + c) if s+c != 0 else 0

        # Nouns
        num_subj_noun = np.count_nonzero(subj_pos == "NOUN") if s != 0 else 0
        perc_subj_noun = num_subj_noun / s if s != 0 else 0
        num_cont_noun = np.count_nonzero(cont_pos == "NOUN") if c != 0 else 0
        perc_cont_noun = num_cont_noun / c if c != 0 else 0
        num_total_noun = num_subj_noun + num_cont_noun if s+c != 0 else 0
        perc_total_noun = num_total_noun / (s + c) if s+c != 0 else 0

        # Numbers
        num_subj_num = np.count_nonzero(subj_pos == "NUM") if s != 0 else 0
        perc_subj_num = num_subj_num / s if s != 0 else 0
        num_cont_num = np.count_nonzero(cont_pos == "NUM") if c != 0 else 0
        perc_cont_num = num_cont_num / c if c != 0 else 0
        num_total_num = num_subj_num + num_cont_num if s+c != 0 else 0
        perc_total_num = num_total_num / (s + c) if s+c != 0 else 0

        # Proper Nouns
        num_subj_propn = np.count_nonzero(subj_pos == "PROPN") if s != 0 else 0
        perc_subj_propn = num_subj_propn / s if s != 0 else 0
        num_cont_propn = np.count_nonzero(cont_pos == "PROPN") if c != 0 else 0
        perc_cont_propn = num_cont_propn / c if c != 0 else 0
        num_total_propn = num_subj_propn + num_cont_propn if s+c != 0 else 0
        perc_total_propn = num_total_propn / (s + c) if s+c != 0 else 0

        # Symbols
        num_subj_sym = np.count_nonzero(subj_pos == "SYM") if s != 0 else 0
        perc_subj_sym = num_subj_sym / s if s != 0 else 0
        num_cont_sym = np.count_nonzero(cont_pos == "SYM") if c != 0 else 0
        perc_cont_sym = num_cont_sym / c if c != 0 else 0
        num_total_sym = num_subj_sym + num_cont_sym if s+c != 0 else 0
        perc_total_sym = num_total_sym / (s + c) if s+c != 0 else 0

        # Verbs
        num_subj_verb = np.count_nonzero(subj_pos == "VERB") if s != 0 else 0
        perc_subj_verb = num_subj_verb / s if s != 0 else 0
        num_cont_verb = np.count_nonzero(cont_pos == "VERB") if c != 0 else 0
        perc_cont_verb = num_cont_verb / c if c != 0 else 0
        num_total_verb = num_subj_verb + num_cont_verb if s+c != 0 else 0
        perc_total_verb = num_total_verb / (s + c) if s+c != 0 else 0

        # Interjections
        num_subj_intj = np.count_nonzero(subj_pos == "INTJ") if s != 0 else 0
        perc_subj_intj = num_subj_intj / s if s != 0 else 0
        num_cont_intj = np.count_nonzero(cont_pos == "INTJ") if c != 0 else 0
        perc_cont_intj = num_cont_intj / c if c != 0 else 0
        num_total_intj = num_subj_intj + num_cont_intj if s+c != 0 else 0
        perc_total_intj = num_total_intj / (s + c) if s+c != 0 else 0

        # Add to data frame
        added_dict.append({# Adjectives
                            "num_subj_adjs": num_subj_adjs,
                           "perc_subj_adjs": perc_subj_adjs,
                           "num_cont_adjs": num_cont_adjs,
                           "perc_cont_adjs": perc_cont_adjs,
                           "num_total_adjs": num_total_adjs,
                           "perc_total_adjs": perc_total_adjs,

                            # Adverbs
                           "num_subj_advs": num_subj_advs,
                            "perc_subj_advs": perc_subj_advs,
                            "num_cont_advs": num_cont_advs,
                            "perc_cont_advs": perc_cont_advs,
                            "num_total_advs": num_total_advs,
                            "perc_total_advs": perc_total_advs,

                           # Nouns
                           "num_subj_noun" : num_subj_noun,
                            "perc_subj_noun": perc_subj_noun,
                            "num_cont_noun": num_cont_noun,
                            "perc_cont_noun": perc_cont_noun,
                            "num_total_noun": num_total_noun,
                            "perc_total_noun": perc_total_noun,

                            # Numbers
                            "num_subj_num": num_subj_num,
                            "perc_subj_num": perc_subj_num,
                            "num_cont_num": num_cont_num,
                            "perc_cont_num": perc_cont_num,
                            "num_total_num": num_total_num,
                            "perc_total_num": perc_total_num,

                            # Proper Nouns
                            "num_subj_propn": num_subj_propn,
                            "perc_subj_propn": perc_subj_propn,
                            "num_cont_propn": num_cont_propn,
                            "perc_cont_propn": perc_cont_propn,
                            "num_total_propn": num_total_propn,
                            "perc_total_propn": perc_total_propn,

                            # Symbols
                            "num_subj_sym": num_subj_sym,
                            "perc_subj_sym": perc_subj_sym,
                            "num_cont_sym": num_cont_sym,
                            "perc_cont_sym": perc_cont_sym,
                            "num_total_sym": num_total_sym,
                            "perc_total_sym": perc_total_sym,

                            # Verbs
                            "num_subj_verb": num_subj_verb,
                            "perc_subj_verb": perc_subj_verb,
                            "num_cont_verb": num_cont_verb,
                            "perc_cont_verb": perc_cont_verb,
                            "num_total_verb": num_total_verb,
                            "perc_total_verb": perc_total_verb,

                            # Interjections
                            "num_subj_intj": num_subj_intj,
                            "perc_subj_intj": perc_subj_intj,
                            "num_cont_intj": num_cont_intj,
                            "perc_cont_intj": perc_cont_intj,
                            "num_total_intj": num_total_intj,
                            "perc_total_intj": perc_total_intj
                             })

    subject = pd.DataFrame(added_dict)
    subject.to_csv("new_cols.csv", header=True)


    pass


def chisquared():
    """Get my chisquared test"""

    df = pd.read_csv("means.csv")

    print(df[df["Label"]==0])

    spam_means = df[df["Label"]==0]["total_mean"]
    ham_means = df[df["Label"]==1]["total_mean"]

    spam_mean_perc = df[df["Label"] == 0]["total_mean_perc"]
    ham_mean_perc = df[df["Label"] == 1]["total_mean_perc"]

    spam_subj_mean = df[df["Label"]==0]["subj_mean"]
    ham_subj_mean = df[df["Label"]==1]["subj_mean"]
    spam_subj_perc = df[df["Label"]==0]["subj_mean_perc"]
    ham_subj_perc = df[df["Label"]==1]["subj_mean_perc"]

    print(np.array([spam_means, ham_means]))

    means = np.array([spam_means, ham_means])

    # Run chisquare test
    stat, p, dof, ex = stats.chi2_contingency(means.T)

    stat, p, dof, ex = stats.chi2_contingency(np.array([spam_mean_perc, ham_mean_perc]))

    stat, p, dof, ex = stats.chi2_contingency(np.array([spam_subj_mean, ham_subj_mean]))
    print("contigency p: ", p, dof)
    stat, p, dof, ex = stats.chi2_contingency(np.array([spam_subj_perc, ham_subj_perc]).T)
    print("contigency p: ", p, dof)

    pass



if __name__ == "__main__":
    # create_csv()
    # testNB()
    # testRF()
    # testVectorizerRF()
    # testVectorizerNB()
    # get_pos()
    # get_pos2()
    chisquared()


    pass