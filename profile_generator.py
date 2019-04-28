import  random

def generate_profiles(n, topics):
    profiles = {}
    for n in range(n):
        number_of_topics = random.randint(1,len(topics))
        profiles[n] = random.sample(topics, number_of_topics)
    return profiles
