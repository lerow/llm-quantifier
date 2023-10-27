import nltk
import inflect
import random

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

def is_plural_wn(noun):
    wnl = nltk.stem.WordNetLemmatizer()
    lemma = wnl.lemmatize(noun, 'n')
    plural = True if noun is not lemma else False
    return plural


def get_plural(singular_noun):
    p = inflect.engine()
    return p.plural(singular_noun)


def get_freq_eng_nouns():
    web_freq_file = open("./data/engl-word-freq/eng-com_web-public_2018_100K/eng-com_web-public_2018_100K-words.txt", "r")
    lines = web_freq_file.readlines()
    web_freq_file.close()

    out_file = open("./data/engl-word-freq/eng-web-nouns-2018.txt", "w")

    for line in lines:
        if line is None or len(line) < 3:
            break

        line = line.split()
        rank = line[0]
        word = line[1]
        freq = line[2]

        if len(word) < 3:
            continue

        tokenized = nltk.word_tokenize(word)
        tag = nltk.pos_tag(tokenized)[0][1]

        # NN, NNP, NNS
        noun_tags = {"NN", "NNP", "NNS"}
        if tag in noun_tags:
            if is_plural_wn(word) or is_plural_wn(get_plural(word)):
                s = word + " " + freq + "\n"
                out_file.write(s)


    out_file.close()


def sample_nouns():
    freq_file = open("./data/engl-word-freq/eng-web-nouns-2018.txt", "r")
    lines = freq_file.readlines()
    freq_file.close()

    out_sample_file = open("./data/engl-word-freq/sample-eng-nouns.txt", "w")

    start_index = 0
    interval_len = 20000 // 10

    for i in range(1, 11):
        samples = random.sample(lines[start_index : interval_len * i], 20)
        start_index = interval_len * i

        for sample in samples:
            out_sample_file.write(sample)
            out_sample_file.write("\n")


    out_sample_file.close()


# get_freq_eng_nouns()

sample_nouns()
