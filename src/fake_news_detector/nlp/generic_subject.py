import spacy
nlp = spacy.load('en')
sent = "I shot an elephant"
doc=nlp(sent)

sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj") ]

print(sub_toks) 
