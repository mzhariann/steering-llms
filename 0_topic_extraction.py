from datasets import load_dataset
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
import pickle

dataset = load_dataset("blog_authorship_corpus", split="train")
dataset_test = load_dataset("blog_authorship_corpus", split="validation")
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

data_words = list(sent_to_words(dataset['text']))
data_words = remove_stopwords(data_words)

test_data_words = list(sent_to_words(dataset_test['text']))
test_data_words = remove_stopwords(test_data_words)

# Create Dictionary
dictionary = corpora.Dictionary(data_words)
dictionary.filter_extremes(no_below=20, no_above=0.1)
# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [dictionary.doc2bow(text) for text in texts]
test_corpus = [dictionary.doc2bow(text) for text in test_data_words]

num_topics = 50
chunksize = 2000
passes = 20
iterations = 1000
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

'''
lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)
'''

#lda_model.save('/ivi/ilps/personal/mariann/ikea/lda.model')
#dictionary.save('/ivi/ilps/personal/mariann/ikea/lda.dict')

lda_model = gensim.models.LdaModel.load('/ivi/ilps/personal/mariann/ikea/lda.model')

#doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
test_doc_topics = [lda_model.get_document_topics(doc) for doc in test_corpus]

'''
with open('/ivi/ilps/personal/mariann/ikea/doc_topics.pkl', 'wb') as f:
    pickle.dump(doc_topics,f)
'''
with open('/ivi/ilps/personal/mariann/ikea/test_doc_topics.pkl', 'wb') as f:
    pickle.dump(test_doc_topics,f)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics(num_topics))

