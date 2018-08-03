"""Name: article_classifier
Description: Generic classifier. Tests types of classifiers and generates benchmarking plot.
            Available choices of classifiers are Stochastic Gradient Descent(SGD) and Passive
            Agreesive (PA) Classifiers.

Usage:
        ln[1] classifier = ArticleClassifier(plot=True)
        ln[2] classifier.extract_features(df)
        ln[3] classifier.test_classifier()
        ln[4] classifier.select_sgdclf(store=True)

Author: Emmanuel Nunez
Email: eanunez85@gmail.com

Credits to scikit-learn.org"""

import os
import logging
from time import time
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics


class ArticleClassifier(object):
    """Classifies articles"""

    feature_names = None
    X_train, X_test, y_train, y_test = [], [], [], []

    def __init__(self, use_hashing=False, n_features=int(2**16), select_chi2=None, print_top10=False,
                 print_report=False, print_cm=False, plot=False):
        """Constructor"""
        self.use_hashing = use_hashing
        self.n_features = n_features
        self.select_chi2 = select_chi2
        self.print_top10 = print_top10
        self.print_report = print_report
        self.print_cm = print_cm
        self.plot = plot

    def test_classifier(self):
        """test classifier"""

        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        results = []
        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                (Perceptron(n_iter=50), "Perceptron"),
                (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (RandomForestClassifier(n_estimators=100), "Random forest")):
            print('=' * 80)
            print(name)
            results.append(self.benchmark(clf))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            results.append(self.benchmark(LinearSVC(penalty=penalty, dual=False, tol=1e-3)))

            # Train SGD model
            results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty)))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(self.benchmark(NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(self.benchmark(MultinomialNB(alpha=.01)))
        results.append(self.benchmark(BernoulliNB(alpha=.01)))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(self.benchmark(Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                            tol=1e-3))),
            ('classification', LinearSVC(penalty="l2"))])))

        if self.plot:
            # GENERATE PLOT
            self.plot_clf(results)
        return results

    def benchmark(self, clf):
        """Benchmarks classifiers"""

        def trim(s):
            """Trim string to fit on terminal (assuming 80-column display)"""
            return s if len(s) <= 80 else s[:77] + "..."

        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(self.X_train, self.y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(self.X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(self.y_test, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            if self.print_top10 and self.feature_names is not None:
                print("top 10 keywords per class:")
                for i, label in enumerate(self.y_train):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s" % (label, " ".join(self.feature_names[top10]))))
            print()

        if self.print_report:
            print("classification report:")
            print(metrics.classification_report(self.y_test, pred,
                                                target_names=self.y_test))

        if self.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(self.y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time

    def select_sgdclf(self, penalty="elasticnet", store=False):
        """Selects SGDClassifier"""

        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        def trim(s):
            """Trim string to fit on terminal (assuming 80-column display)"""
            return s if len(s) <= 80 else s[:77] + "..."

        print('=' * 80)
        print("%s penalty" % penalty.upper())

        # Train SGD model
        clf = SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty)

        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(self.X_train, self.y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(self.X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(self.y_test, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            if self.print_top10 and self.feature_names is not None:
                print("top 10 keywords per class:")
                for i, label in enumerate(self.y_train):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s" % (label, " ".join(self.feature_names[top10]))))
            print()

        if self.print_report:
            print("classification report:")
            print(metrics.classification_report(self.y_test, pred,
                                                target_names=self.y_test))

        if self.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(self.y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]

        if store:
            base, f_name = os.path.split(os.path.abspath('article_classifier.py'))
            filename = base + '/' + 'sgd_clf.pkl'
            joblib.dump(clf, filename)
            print('Saved ', os.path.abspath(filename))

        return clf_descr, score, train_time, test_time

    def select_paclf(self, store=False):
        """Selects PassiveAgressiveClassifier"""

        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        def trim(s):
            """Trim string to fit on terminal (assuming 80-column display)"""
            return s if len(s) <= 80 else s[:77] + "..."

        print('=' * 80)

        clf = PassiveAggressiveClassifier(n_iter=50)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(self.X_train, self.y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(self.X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(self.y_test, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            if self.print_top10 and self.feature_names is not None:
                print("top 10 keywords per class:")
                for i, label in enumerate(self.y_train):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s" % (label, " ".join(self.feature_names[top10]))))
            print()

        if self.print_report:
            print("classification report:")
            print(metrics.classification_report(self.y_test, pred,
                                                target_names=self.y_test))

        if self.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(self.y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]

        if store:
            base, f_name = os.path.split(os.path.abspath('article_classifier.py'))
            filename = base + '/' + 'pa_clf.pkl'
            joblib.dump(clf, filename)
            print('Saved ', os.path.abspath(filename))

        return clf_descr, score, train_time, test_time

    def select_multinominalnb(self, alpha=.01, store=False):
        """Select Multinominal NB model"""

        # Display progress logs on stdout
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')

        def trim(s):
            """Trim string to fit on terminal (assuming 80-column display)"""
            return s if len(s) <= 80 else s[:77] + "..."

        print('=' * 80)

        clf = MultinomialNB(alpha=alpha)

        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(self.X_train, self.y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(self.X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(self.y_test, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            if self.print_top10 and self.feature_names is not None:
                print("top 10 keywords per class:")
                for i, label in enumerate(self.y_train):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s" % (label, " ".join(self.feature_names[top10]))))
            print()

        if self.print_report:
            print("classification report:")
            print(metrics.classification_report(self.y_test, pred,
                                                target_names=self.y_test))

        if self.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(self.y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]

        if store:
            base, f_name = os.path.split(os.path.abspath('article_classifier.py'))
            filename = base + '/' + 'mnb_clf.pkl'
            joblib.dump(clf, filename)
            print('Saved ', os.path.abspath(filename))

        return clf_descr, score, train_time, test_time

    def extract_features(self, df, store_vect=True):
        """Extracts features

        :param df: dataframe with column keys, 'input' and 'label',
                    where 'input' are the contents of articles and 'label' are the desired results/targets
        :param store_vect: store vectorizer to pickle
        :return x_train, x_text, y_train, y_test:
        vectorized sets of train and test with .8:.2 ratio, respectively"""

        # SPLIT TRAIN AND TEST
        print("Extracting features from the training data using a sparse vectorizer")
        t0 = time()
        y_train = df['label'][:int(len(df['label']) * .8)]
        y_test = df['label'][int(len(df['label']) * .8):]

        print('Training set: ', len(y_train))
        print('Testing set: ', len(y_test))

        if self.use_hashing and self.n_features:
            vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                           n_features=self.n_features)
            x_train = vectorizer.transform(df['input'][:int(len(df['input']) * .8)])
        else:
            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
            x_train = vectorizer.fit_transform(df['input'][:int(len(df['input']) * .8)])

        try:
            duration = time() - t0
            print("done in %fs at %0.3fMB/s" % (duration, df['input'][:int(len(df['input']) * .8)].size / duration))
            print("n_samples: %d, n_features: %d" % x_train.shape)
            print()
        except ZeroDivisionError as err:
            print(err)

        print("Extracting features from the test data using the same vectorizer")
        t0 = time()
        x_test = vectorizer.transform(df['input'][int(len(df['input']) * .8):])
        duration = time() - t0
        try:
            print("done in %fs at %0.3fMB/s" % (duration, df['input'][int(len(df['input']) * .8):].size / duration))
            print("n_samples: %d, n_features: %d" % x_test.shape)
            print()
        except ZeroDivisionError as err:
            print(err)

        # mapping from integer feature name to original token string
        if self.use_hashing:
            feature_names = None
        else:
            feature_names = vectorizer.get_feature_names()

        if self.select_chi2:
            print("Extracting %d best features by a chi-squared test" % self.select_chi2)
            t0 = time()
            ch2 = SelectKBest(chi2, k=self.select_chi2)
            x_train = ch2.fit_transform(x_train, y_train)
            x_test = ch2.transform(x_test)
            if feature_names:
                # keep selected feature names
                self.feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
            print("done in %fs" % (time() - t0))
            print()

        if feature_names:
            self.feature_names = np.asarray(feature_names)

        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        if store_vect:
            base, f_name = os.path.split(os.path.abspath('article_classifier.py'))
            filename = base + '/' + 'vectorizer.pkl'
            joblib.dump(vectorizer, filename)
            print('Saved ', os.path.abspath(filename))

        return x_train, x_test, y_train, y_test

    @staticmethod
    def plot_clf(results):
        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]

        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        plt.barh(indices + .3, training_time, .2, label="training time",
                 color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        plt.show()
