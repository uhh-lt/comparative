import pandas as pd
import spacy

nlp = spacy.load('en')


if __name__ == '__main__':
    for sentence in list(["I think that this computer is crap"]):
        doc = nlp(sentence)
        print('## Named Entities')
        for entity in doc.ents:
            print(entity.label_, entity)

        print('\n\n## Tags')
        for token in doc:
            print('{} | POS {} | Lemma {} | Tag {}'.format(token, token.pos_, token.lemma_, token.tag_))

        print('\n\n## Dependency')
        for np in doc.noun_chunks:
            print(np.text, np.root.text, np.root.dep_, np.root.head.text)

        print('\n\n## Sentiment')
        print(doc.sentiment)

        print('Misc')
        for token in doc:
            print('{} | Log Prob {} | Stopword {}'.format(token,  token.prob, token.is_stop))
