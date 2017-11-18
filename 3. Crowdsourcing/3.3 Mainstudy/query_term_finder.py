from jobimtextapi import jobimtext

jb = jobimtext.JoBimText()

resp = jb.similar('worse', holingtype='trigram')
for s in reversed(resp.by_score(min_score=200)):
    print(s)

has, sense = jb.senses('better',holingtype='trigram').has_sense('worse')
print(has)
print(sense)

print(jb.similar_score('Pepsi', None, 'Cola', None,holingtype='trigram').score)
    