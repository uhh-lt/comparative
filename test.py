a = ['the', 'a', 'okay']
b = ['the']


def remove_stopwords(lst, stopwords):
    return [x for x in lst if x not in stopwords]

print(remove_stopwords(a,b))