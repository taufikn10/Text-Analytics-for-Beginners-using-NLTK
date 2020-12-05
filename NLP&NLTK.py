#Loading NLTK
import nltk

#1. Tokenization
# Sentence tokenizer memecah paragraf teks menjadi kalimat
from nltk.tokenize import sent_tokenize
text="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)

#2. World Tokenization
# Word tokenizer memecah kalimat menjadi kata
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

#3. Frequency Distribution
# Menghitung frekuensi kemunculan kata
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)
fdist.most_common(2)
# Frequency Distribution Plot
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()

#4. Stopwords
# Kata yang bukan kata utama, seperti : yang, ini,itu
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)

#5. removing stopwords
# Menghapus kata-kata pendukung
filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)

#6. Stemming
# Stemming adalah proses normalisasi linguistik, yang mereduksi kata dengan memotong imbuhan
from nltk.stem import PorterStemmer

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

#7. Stemming and Lemmatization
# Lemmatisasi mereduksi kata menjadi kata dasarnya

#Lexicon Normalization
#performing stemming and Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))

#8. POS Tagging
# mengidentifikasi kelompok tata bahasa dari kata tertentu
sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nltk.word_tokenize(sent)
print(tokens)

nltk.pos_tag(tokens)

#9. Parse Tree
import nltk

from nltk.tokenize import word_tokenize

text = "I shot an elephant in my pajamas"
tokenized_word=word_tokenize(text)
print(tokenized_word)

groucho_grammar = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")

parser = nltk.ChartParser(groucho_grammar)
for tree in parser.parse(tokenized_word):
  print(tree)
  tree.draw()