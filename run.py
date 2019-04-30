import sys
import pickle
from lib.data import generate_profiles
from sklearn.metrics import accuracy_score

article = None
try:
    with open(sys.argv[1], 'r') as file:
        article = file.read().replace('\n', ' ')
except IndexError:
    print('Please provide path to file as command line argument.')

if article is not None:
    n = 5
    topics = [
        'business',
        'entertainment',
        'politics',
        'sport',
        'tech',
    ]
    profiles = generate_profiles(n, topics)
    print(f'Generated {n} profiles.')
    print(profiles)

    tfidf_model = pickle.load(open('models/tfidf_model.pickle', 'rb'))
    result = tfidf_model.predict(article)
    print('Article classified as category:', result)

    for u, topics in profiles.items():
        if result in topics:
            print('Article will be delivered to user:', u)
