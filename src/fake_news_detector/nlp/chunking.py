from nltk.parse.corenlp import CoreNLPParser
import tkinter

def standford_parse_tree(sentence):
    parser = CoreNLPParser()
    return next(parser.raw_parse(sentence))

def split_by_conj(sentence):
    return