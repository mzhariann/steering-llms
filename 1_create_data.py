from datasets import load_dataset
import pickle
import pandas as pd

dataset = load_dataset("blog_authorship_corpus", split="train")
#dataset = load_dataset("blog_authorship_corpus", split="validation")


with open('/ivi/ilps/personal/mariann/ikea/doc_topics.pkl', 'rb') as f:
    doc_topics = pickle.load(f)

print(len(dataset))
print(len(doc_topics))
lda_topics = pd.read_csv('topics.csv')
lda_topics_dict = dict(zip(lda_topics.topic_id,lda_topics.topic_text))

labels = []
texts = []
prompts = []
profiles = []
for i,doc_topic in enumerate(doc_topics):
    if i%1000 == 0:
        print(i)
    top_doc_topic = sorted(doc_topic, key=lambda x: x[1], reverse=True)[0]
    if top_doc_topic[0] in lda_topics_dict:# and top_doc_topic[1] >= 0.2:
        label = dataset['text'][i]
        profile = dataset['gender'][i] + ' ' + str(dataset['age'][i]) + ' ' + dataset['job'][i] + ' ' + dataset['horoscope'][i]
        prompt = lda_topics_dict[top_doc_topic[0]]
        prompts.append(prompt)
        profiles.append(profile)
        labels.append(label)
        texts.append('prompt: '+prompt+' | profile: '+profile)

print('profiles', len(set(profiles)))
print('prompts', len(set(prompts)))
data_df = pd.DataFrame([texts,labels],index=['text','label']).T
print(data_df.shape)
data_df.to_csv('/ivi/ilps/personal/mariann/ikea/steer_dataset.csv')
