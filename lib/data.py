import os
import  random

def generate_profiles(n, topics):
    profiles = {}
    for n in range(n):
        number_of_topics = random.randint(1,len(topics))
        profiles[n] = random.sample(topics, number_of_topics)
    return profiles

def get_docs(path):
    docs = []
    for r, d, f in os.walk(path):
        if len(d) == 0: # reads only directories without subdirectories
            for file_name in f:
                file_path = os.path.join(r, file_name)
                with open(file_path, 'r') as file:
                    text = ''
                    for line in file:
                        text = text+line
                    docs.append(text)
    return docs
