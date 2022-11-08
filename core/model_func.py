

def transform_text(text : str, nlp, stopwords):
    return [(word.lemma_, word.pos_) for word in nlp(text) if word.text not in stopwords]


def tokenizer(x):
    return [ret[0] for ret in transform_text(x)]
