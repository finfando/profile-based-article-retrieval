import sys
import pickle
from lib.data import generate_profiles
from sklearn.metrics import accuracy_score

query = None
try:
    article = sys.argv[1]
except IndexError:
    print('Please provide query as command line argument.')

if article is not None:
    # n = 5
    # topics = [
    #     'business',
    #     'entertainment',
    #     'politics',
    #     'sport',
    #     'tech',
    # ]
    # profiles = generate_profiles(n, topics)
    # print(f'Generated {n} profiles.')
    # print(profiles)

    print(article)
    
    tfidf_model = pickle.load(open('models/tfidf_model.pickle', 'rb'))
    ranking = tfidf_model.get_ranking(article)
    category = tfidf_model.predict(article)
    print('Query classified as category:', category)

    print('Most relevant articles from index:')

    for r in ranking[:3]:
        print("[ Score = " + "%.3f" % round(r[1],3) + "] ")
        print(tfidf_model.train_set[r[0]]['text'])

    # for u, topics in profiles.items():
    #     if result in topics:
    #         print('Article will be delivered to user:', u)
