# Generated from /Users/fresh/Desktop/Universita/VisioneArtificiale/Progetto/omr-img2midi/data/semantic.g4 by ANTLR 4.13.1
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,0,5,34,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,1,0,1,0,1,
        1,1,1,5,1,16,8,1,10,1,12,1,19,9,1,1,1,1,1,1,2,4,2,24,8,2,11,2,12,
        2,25,1,3,1,3,1,4,4,4,31,8,4,11,4,12,4,32,0,0,5,1,1,3,2,5,3,7,4,9,
        5,1,0,3,2,0,10,10,13,13,4,0,35,35,46,57,65,90,97,122,3,0,9,10,13,
        13,32,32,36,0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,0,0,0,0,7,1,0,0,0,0,9,
        1,0,0,0,1,11,1,0,0,0,3,13,1,0,0,0,5,23,1,0,0,0,7,27,1,0,0,0,9,30,
        1,0,0,0,11,12,5,95,0,0,12,2,1,0,0,0,13,17,5,37,0,0,14,16,8,0,0,0,
        15,14,1,0,0,0,16,19,1,0,0,0,17,15,1,0,0,0,17,18,1,0,0,0,18,20,1,
        0,0,0,19,17,1,0,0,0,20,21,6,1,0,0,21,4,1,0,0,0,22,24,7,1,0,0,23,
        22,1,0,0,0,24,25,1,0,0,0,25,23,1,0,0,0,25,26,1,0,0,0,26,6,1,0,0,
        0,27,28,5,45,0,0,28,8,1,0,0,0,29,31,7,2,0,0,30,29,1,0,0,0,31,32,
        1,0,0,0,32,30,1,0,0,0,32,33,1,0,0,0,33,10,1,0,0,0,4,0,17,25,32,1,
        6,0,0
    ]

class semanticLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    COMMENT = 2
    STRING = 3
    DASH = 4
    WHITESPACE = 5

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'_'", "'-'" ]

    symbolicNames = [ "<INVALID>",
            "COMMENT", "STRING", "DASH", "WHITESPACE" ]

    ruleNames = [ "T__0", "COMMENT", "STRING", "DASH", "WHITESPACE" ]

    grammarFileName = "semantic.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


