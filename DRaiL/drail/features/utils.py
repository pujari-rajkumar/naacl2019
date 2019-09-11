import numpy as np
import sys


def embeddings_dictionary(embedding_file, vocabulary=None, id2int=False):
    '''
    Helper function to build an embeddings dictionary
    from a bin file and a set of vocabulary of words

    Returns the number of embeddings extracted
    the size of the embeddings
    and the key-value mapping from word to vector
    '''
    print "loading embeddings", embedding_file, "..."
    dictionary = {}
    with open(embedding_file, 'r') as f:
        header = f.readline()
        #print header
        num, size = map(int, header.split())
        #print num, size, vocabulary
        bin_len = np.dtype('float32').itemsize * size
        i = 0
        for line in range(num):
            char = f.read(1); word = ""
            while char != ' ':
                word += char
                char = f.read(1)

            vector = np.fromstring(f.read(bin_len), dtype='float32')
            vector = map(float, list(vector))
            #x = f.read(1)
            if vocabulary is None:
                if not id2int:
                    dictionary[word.lower()] = vector
                else:
                    dictionary[int(word.lower())] = vector
            elif word.lower() in vocabulary:
                if not id2int:
                    dictionary[word.lower()] = vector
                else:
                    dictionary[int(word.lower())] = vector

            i += 1
            '''
            this is here for stopping early when debugging
            if i == 100:
                break
            '''
            #sys.stdout.write("\r\t{0}%".format((i * 100) / num))
    #sys.stdout.write('\n')
    return (len(dictionary), size, dictionary)

def embeddings_dictionary_txt(embedding_file, vocabulary=None):
    '''
    Helper function to build an embeddings dictionary
    from a bin file and a set of vocabulary of words
    (Built to work on Google's w2v embeddings)

    Returns the number of embeddings extracted
    the size of the embeddings
    and the key-value mapping from word to vector
    '''
    print "loading embeddings", embedding_file, "..."
    dictionary = {}
    with open(embedding_file, 'r') as f:
        for line in f:
            splits = line.strip().split()
            word = splits[0]
            vector = map(float, splits[1:])
            size = len(vector)
            if vocabulary is None:
                dictionary[word.lower()] = vector
            elif word.lower() in vocabulary:
                dictionary[word.lower()] = vector

    return (len(dictionary), size, dictionary)

def onehot_dictionary(vocabulary):
    '''
    Helper function to build a one hot dictionary
    from a set of vocabulary of words

    Returns the number of elements extracted
    the size of the vector
    and the key-value mapping from word to vector
    '''
    dictionary = {}
    size = len(vocabulary)
    num = len(vocabulary)
    for i, v in enumerate(vocabulary):
        dictionary[v] = np.zeros(size)
        dictionary[v] = map(float, list(dictionary[v]))
        dictionary[v][i] = 1
    return (num, size, dictionary)

def idx_dictionary(vocabulary):
    dictionary = {}
    num = len(vocabulary)
    for i, v in enumerate(vocabulary):
        dictionary[v] = i
    return (num, dictionary)

def vec_average(vectors):
    size = len(vectors[0])

    avgVec = np.zeros(size)
    avgVec = map(float, list(avgVec))
    for vector in vectors:
        if (len(vector) != size):
            print("         ==================== size difference in vectors")
            return None
        avgVec += vector
    avgVec /= len(vectors)

    return avgVec
