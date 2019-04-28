import pickle
from lib.data import generate_profiles
from sklearn.metrics import accuracy_score

n = 5
topics = [
    'business',
    'entertainment',
    'politics',
    'sport',
    'tech',
]

profiles = generate_profiles(n, topics)
tfidf_model = pickle.load(open('models/tfidf_model.pickle', 'rb'))

print(f'Generated {n} profiles. Started looking for relevant relevant articles. Number of articles to assign: {len(tfidf_model.test_set)}.')
print(profiles)

tfidf_model = pickle.load(open('models/tfidf_model.pickle', 'rb'))
true_labels = [a['category'] for a in tfidf_model.test_set]
predictions = []

# assign articles to categories
articles_assigned = {}
for t in topics:
    articles_assigned[t] = []
for article in tfidf_model.test_set:
    title = article['text'].split('\n')[0]
    result = tfidf_model.predict(article['text'])
    articles_assigned[result].append(title)
    predictions.append(result)

# get relevant articles for each user
for u, topics in profiles.items():
    print('User:', u)
    print('Topics:', topics)
    articles = []
    for t in topics:
        articles = articles+articles_assigned[t]
    print(articles)

print('Accuracy score:', accuracy_score(true_labels, predictions))
