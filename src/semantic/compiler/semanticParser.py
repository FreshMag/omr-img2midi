# Generated from /Users/fresh/Desktop/Universita/VisioneArtificiale/Progetto/omr-img2midi/data/semantic.g4 by ANTLR 4.13.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,5,39,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,1,0,1,0,1,1,5,1,
        14,8,1,10,1,12,1,17,9,1,1,1,1,1,1,1,5,1,22,8,1,10,1,12,1,25,9,1,
        1,2,1,2,1,2,3,2,30,8,2,1,3,1,3,1,4,4,4,35,8,4,11,4,12,4,36,1,4,0,
        0,5,0,2,4,6,8,0,1,2,0,1,1,3,3,37,0,10,1,0,0,0,2,23,1,0,0,0,4,26,
        1,0,0,0,6,31,1,0,0,0,8,34,1,0,0,0,10,11,3,2,1,0,11,1,1,0,0,0,12,
        14,5,5,0,0,13,12,1,0,0,0,14,17,1,0,0,0,15,13,1,0,0,0,15,16,1,0,0,
        0,16,18,1,0,0,0,17,15,1,0,0,0,18,19,3,4,2,0,19,20,5,5,0,0,20,22,
        1,0,0,0,21,15,1,0,0,0,22,25,1,0,0,0,23,21,1,0,0,0,23,24,1,0,0,0,
        24,3,1,0,0,0,25,23,1,0,0,0,26,29,3,6,3,0,27,28,5,4,0,0,28,30,3,8,
        4,0,29,27,1,0,0,0,29,30,1,0,0,0,30,5,1,0,0,0,31,32,5,3,0,0,32,7,
        1,0,0,0,33,35,7,0,0,0,34,33,1,0,0,0,35,36,1,0,0,0,36,34,1,0,0,0,
        36,37,1,0,0,0,37,9,1,0,0,0,4,15,23,29,36
    ]

class semanticParser ( Parser ):

    grammarFileName = "semantic.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'_'", "<INVALID>", "<INVALID>", "'-'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "COMMENT", "STRING", "DASH", 
                      "WHITESPACE" ]

    RULE_file = 0
    RULE_semantics = 1
    RULE_symbol = 2
    RULE_type = 3
    RULE_identifier = 4

    ruleNames =  [ "file", "semantics", "symbol", "type", "identifier" ]

    EOF = Token.EOF
    T__0=1
    COMMENT=2
    STRING=3
    DASH=4
    WHITESPACE=5

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class FileContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def semantics(self):
            return self.getTypedRuleContext(semanticParser.SemanticsContext,0)


        def getRuleIndex(self):
            return semanticParser.RULE_file

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFile" ):
                listener.enterFile(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFile" ):
                listener.exitFile(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFile" ):
                return visitor.visitFile(self)
            else:
                return visitor.visitChildren(self)




    def file_(self):

        localctx = semanticParser.FileContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_file)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 10
            self.semantics()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SemanticsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def symbol(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(semanticParser.SymbolContext)
            else:
                return self.getTypedRuleContext(semanticParser.SymbolContext,i)


        def WHITESPACE(self, i:int=None):
            if i is None:
                return self.getTokens(semanticParser.WHITESPACE)
            else:
                return self.getToken(semanticParser.WHITESPACE, i)

        def getRuleIndex(self):
            return semanticParser.RULE_semantics

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSemantics" ):
                listener.enterSemantics(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSemantics" ):
                listener.exitSemantics(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSemantics" ):
                return visitor.visitSemantics(self)
            else:
                return visitor.visitChildren(self)




    def semantics(self):

        localctx = semanticParser.SemanticsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_semantics)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 23
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==3 or _la==5:
                self.state = 15
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==5:
                    self.state = 12
                    self.match(semanticParser.WHITESPACE)
                    self.state = 17
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                self.state = 18
                self.symbol()
                self.state = 19
                self.match(semanticParser.WHITESPACE)
                self.state = 25
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SymbolContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def type_(self):
            return self.getTypedRuleContext(semanticParser.TypeContext,0)


        def DASH(self):
            return self.getToken(semanticParser.DASH, 0)

        def identifier(self):
            return self.getTypedRuleContext(semanticParser.IdentifierContext,0)


        def getRuleIndex(self):
            return semanticParser.RULE_symbol

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSymbol" ):
                listener.enterSymbol(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSymbol" ):
                listener.exitSymbol(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSymbol" ):
                return visitor.visitSymbol(self)
            else:
                return visitor.visitChildren(self)




    def symbol(self):

        localctx = semanticParser.SymbolContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_symbol)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 26
            self.type_()
            self.state = 29
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==4:
                self.state = 27
                self.match(semanticParser.DASH)
                self.state = 28
                self.identifier()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self):
            return self.getToken(semanticParser.STRING, 0)

        def getRuleIndex(self):
            return semanticParser.RULE_type

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterType" ):
                listener.enterType(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitType" ):
                listener.exitType(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitType" ):
                return visitor.visitType(self)
            else:
                return visitor.visitChildren(self)




    def type_(self):

        localctx = semanticParser.TypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_type)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 31
            self.match(semanticParser.STRING)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def STRING(self, i:int=None):
            if i is None:
                return self.getTokens(semanticParser.STRING)
            else:
                return self.getToken(semanticParser.STRING, i)

        def getRuleIndex(self):
            return semanticParser.RULE_identifier

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIdentifier" ):
                listener.enterIdentifier(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIdentifier" ):
                listener.exitIdentifier(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIdentifier" ):
                return visitor.visitIdentifier(self)
            else:
                return visitor.visitChildren(self)




    def identifier(self):

        localctx = semanticParser.IdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_identifier)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 34 
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while True:
                self.state = 33
                _la = self._input.LA(1)
                if not(_la==1 or _la==3):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 36 
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if not (_la==1 or _la==3):
                    break

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





