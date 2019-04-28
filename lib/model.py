#!/usr/bin/python
import os
import random

from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models, similarities
from operator import itemgetter
from collections import defaultdict

class tfidf(object):
    def __init__(self):
        pass

    def get_articles(self, path):
        self.docs = []
        for r, d, f in os.walk(path):
            if len(d) == 0: # reads only directories without subdirectories
                for file_name in f:
                    file_path = os.path.join(r, file_name)
                    with open(file_path, 'r') as file:
                        text = ''
                        for line in file:
                            text = text+line
                        self.docs.append({
                            'category': r.split('\\')[-1],
                            'text': text,
                            })

    def split_data(self):
        random.shuffle(self.docs)
        print('Number of articles:', len(self.docs))
        split_at = int(len(self.docs)*0.9)
        self.train_set = self.docs[:split_at]
        self.test_set = self.docs[split_at:]
        print('Training:', len(self.train_set))
        print('Test:', len(self.test_set))

    def preprocess_document(self, doc):
        stopset = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = wordpunct_tokenize(doc)
        clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]

        # remove rare words
        frequency = defaultdict(int)
        for token in clean:
            frequency[token] += 1
        clean = [token for token in clean if frequency[token] > 1]

        final = [stemmer.stem(word) for word in clean]
        return final

    def get_keyword_to_id_mapping(self):
        print(self.dictionary.token2id)

    def create_dictionary(self, docs):
        pdocs = [self.preprocess_document(doc) for doc in docs]
        self.dictionary = corpora.Dictionary(pdocs)

    def docs2bows(self, corpus, dictionary):
        docs = [self.preprocess_document(d) for d in corpus]
        self.vectors = [dictionary.doc2bow(doc) for doc in docs]

    def train(self):
        docs = [i['text'] for i in self.train_set]
        self.create_dictionary(docs)
        self.docs2bows(docs, self.dictionary)
        self.model = models.TfidfModel(self.vectors)

    def get_ranking(self, article):
        index = similarities.MatrixSimilarity(self.vectors, num_features=len(self.dictionary))
        pq = self.preprocess_document(article)
        vq = self.dictionary.doc2bow(pq)
        qtfidf = self.model[vq]
        sim = index[qtfidf]
        ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
        return ranking

    def predict(self, article):
        ranking = self.get_ranking(article)
        categories = [self.train_set[r[0]]['category'] for r in ranking[:30]]
        freqs = {}
        for c in categories:
            if c in freqs:
                freqs[c] += 1
            else:
                freqs[c] = 1
        result = max(freqs.items(), key=itemgetter(1))[0]
        return result
