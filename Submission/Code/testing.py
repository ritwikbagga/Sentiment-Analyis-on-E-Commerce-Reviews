import string

def preprocess( text):
    """
    Preprocessing of one Review Text
        - convert to lowercase done
        - remove punctuation
        - empty spaces
        - remove 1-letter words
        - split the sentence into words

    Return the split words
    """
    # lower case
    text = text.lower()
    words = text.split()
    puntuation = ""
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    final_words = []
    for word in stripped:  # remove one letter word
        if len(word) > 1:
            final_words.append(word)

    return final_words

text= "hi my NNNNNName is? what isSS it ?? i dont know but i like this!"

print(preprocess(text))