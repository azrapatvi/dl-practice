from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences

max_len=500 

word_index=imdb.get_word_index()

def preprocess_text(text):

    text = text.lower()          
    words = text.split()

    text_indexes=[word_index.get(word,2)+3 for word in words]

    padded_text = pad_sequences([text_indexes], maxlen=max_len)
    
    return padded_text