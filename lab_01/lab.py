import math
import re # for tokenization

N = 30 # given in the task fixed query set

# Load the documents
# Documents structure: _id, title, text, metadata
def load_nfcorpus():

# Tokenizing schema: lowercase → strip punctuation → split on whitespace
def tokenize(text):
    lowercased = text.lower()

    stripped = re.sub(r'[^\w\s]', '', lowercased)

    tokenized = stripped.split()

    return tokenized

def build_inverted_index():

def calculate_tf(t, d):

def calculate_idf(t):


def calculate_tf_idf():
    tf = 0
    df = 0

    idf = math.log(N/df)
    w = tf * idf
    return w

def calculate_bm25(q, d):
    result = 0
    k1 = 1.5
    b = 0.75

    df = calculate_idf(t)

    B = ( (k1 + 1) * f) / (k1 * ( (1 - b) + b * (len / avgdl)) + f)
    idf_clamped = math.log(1 + (N - df +0.5)/(df+0.5))
    
    result += B * idf_clamped
    return result

corpus, queries, qrels = load_nfcorpus()

print(len(corpus))
print(len(queries))
print(corpus[0])
print(queries[0])