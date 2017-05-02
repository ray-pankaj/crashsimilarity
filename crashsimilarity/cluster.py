import sklearn.cluster
import crash_similarity
from pyemd import emd
import time
import gensim
import numpy as np


def create_distance_matrix(dictionary, docset):
    distances = np.zeros((len(dictionary), len(dictionary)), dtype=np.double)
    for j, w in dictionary.items():
        if w in docset:
            distances[:all_distances.shape[1], j] = all_distances[model.wv.vocab[w].index].transpose()

    return distances


def wmdistance(words1, words2):
    dictionary = gensim.corpora.Dictionary(documents=[words1, words2])
    vocab_len = len(dictionary)

    # create bag of words from document
    def create_bow(doc):
        norm_bow = np.zeros(vocab_len, dtype=np.double)
        bow = dictionary.doc2bow(doc)

        for idx, count in bow:
            norm_bow[idx] = count / float(len(doc))

        return norm_bow

    bow1 = create_bow(words1)
    bow2 = create_bow(words2)

    docset = set(words2)
    distances = create_distance_matrix(dictionary, docset)

    assert sum(distances) != 0

    return emd(bow1, bow2, distances)


def wmdutil(X):
    n = len(X)
    for i in range(n):
        for j in range(i + 1, n):
            return wmdistance(X[i], X[j])


start = time.time()

paths = ['../crashsimilarity_data/firefox-crashes-2016-11-09.json.gz']
corpus = crash_similarity.read_corpus(paths)
print len(corpus)
# for i in corpus:
#     print len(i.words)
model = crash_similarity.train_model(corpus)
model.init_sims(replace=True)
total_words_to_test = []
for doc in corpus:
    # print (type(doc), doc)
    # print (doc[0], doc[1])
    # print (doc.words)
    total_words_to_test.append(crash_similarity.preprocess(' | '.join(doc.words)))
print "Data created", time.time() - start
start2 = time.time()
words_to_test_clean = []
all_words = set()
for words_to_test in total_words_to_test:
    x = [w for w in np.unique(words_to_test).tolist() if w in model]
    if len(x) >= 10:
        words_to_test_clean.append(x[:10])
for words in words_to_test_clean:
    for w in words:
        all_words.add(w)
all_distances = np.array(1.0 - np.dot(model.wv.syn0norm, model.wv.syn0norm[[model.wv.vocab[word].index for word in all_words]].transpose()), dtype=np.double)
print all_distances.shape
# print all_words
start3 = time.time()
# print words_to_test_clean
# print all_distances
# for i in range(len(words_to_test_clean)):
#     words_to_test_clean[i] = np.asarray(words_to_test_clean[i])
# words_to_test_clean = np.asarray(words_to_test_clean)
clustermodel = sklearn.cluster.AgglomerativeClustering(linkage='complete', affinity=wmdutil)
clustermodel.fit(words_to_test_clean)
print "Clustering done", time.time() - start3
print "Total time taken", time.time() - start
