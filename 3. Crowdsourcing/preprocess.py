import json
import string
from random import shuffle


def read(file):
    n = []
    lines = []
    cnt = 0
    with open(file) as data:
        lines = [x['_source']['text'] for x in json.load(data)['hits']['hits']]
    return lines


def length_filter(sentence):
    """sentece must be at least 15 chars long"""
    return len(sentence) >= 15 and len(sentence) <= 100

def appears_once(sentence, words):
    """each object should appear exactly once"""
    count = 0
    for word in words:
        count += sentence.lower().count(word.lower())
    return count is len(words)

def not_contains(sentence,words):
    count = 0
    for word in words:
        count += sentence.count(word)
    return count == 0   


def symbols(sentence, max=5):
    """not to much punctuation"""
    count = 0
    for punct in string.punctuation:
        count += sentence.count(punct)
    return count <= max

if __name__ == '__main__':


    objs = [['ruby', 'python'], ['android','iphone'],['cat','dog'], ['car','bicycle'], ['summer', 'winter'], 
    ['bmw','mercedes'], ['wine','beer'], ['usa', 'europe'], ['football','baseball'],['chicken', 'beef']]

    sentences = []
    for pair in objs:
        files = {
            '-'.join(pair): [pair,('better', 'worse', 'superior', 'inferior'), 25],
            '{}-better'.format('-'.join(pair)): [pair,[],75]
        }
        for file, objects in files.items():
            lines = read('data/{}.json'.format(file))
            filtered = [line+'\n' for line in lines if length_filter(line) and appears_once(line, objects[0]) and symbols(line) and not_contains(line, objects[1])]
            shuffle(filtered)
            sentences += filtered[:objects[2]]

    with open('result.txt', 'w') as file:
        file.writelines(sentences)