import nltk
import kenlm
import sys

def load_sentences():
    fn = "./sentences.txt"
    f = open(fn)
    line = f.read()
    f.close()
    words = nltk.tokenize.word_tokenize(line)
    words = [x.lower() for x in words]
    return words

def load_options():
    f = open("./options.txt")
    options = []
    for line in f:
        ll = line.split()
        words = []
        for s in ll:
            word = s.split(".")[-1].lower()
            words.append(word)
        options.append(words)
    f.close()
    return options

def load_answers():
    f = open("./answers.txt")
    line = f.readline().strip()
    f.close()
    return line

def unk_it(words, vocab):
    blank_ids = [str(n) for n in xrange(1,21)]
    for i in xrange(len(words)):
        if words[i] in blank_ids:
            continue
        if not words[i] in vocab:
            words[i] = "UNK"
    return words

def predict(words, options, lm, right_answers):
    ngram = lm.order
    blank_ids = [str(n) for n in xrange(1,21)]
    bid = 0
    answers = ""
    for i in xrange(len(words)):
        if words[i] in blank_ids:
            print "Blank ", words[i]

            max_score = -float('inf')
            choice = -1
            for j in xrange(len(options[bid])):
                option = options[bid][j]
                if not option in lm:
                    option = "UNK"
                seq = words[i - (ngram-1):i] + [option] + words[i+1:i + ngram]
                seq = " ".join(seq)
                score = lm.score(seq,bos=False, eos=False)
                print score, seq
                if score > max_score:
                    max_score = score
                    choice = j
            print "Choose: ", 'ABCD'[choice], "Correct: ", right_answers[bid]
            answers+='ABCD'[choice]
            bid += 1

    return answers

def accuracy(answers, choices):
    n = len(answers)
    c = 0
    for i in xrange(len(answers)):
        if answers[i] == choices[i]:
            c += 1
    return c*1.0/n

if __name__ == "__main__":
    lm = kenlm.Model(sys.argv[1])

    words = load_sentences()
    options = load_options()
    answers = load_answers()

    words = unk_it(words, lm)

    choices = predict(words, options, lm, answers)

    print choices
    print answers

    acc = accuracy(answers,choices)
    print acc
