from antlr4 import *

from .compiler.semanticLexer import semanticLexer
from .compiler.semanticParser import semanticParser
from .compiler.semanticVisitor import semanticVisitor


def parse(text):
    lexer = semanticLexer(InputStream(text))
    stream = CommonTokenStream(lexer)
    parser = semanticParser(stream)
    tree = parser.file_()
    visitor = SemanticVisitor()
    return visitor.visitFile(tree)


class SemanticVisitor(semanticVisitor):
    def __init__(self):
        super(SemanticVisitor, self).__init__()
        self.res = []

    def visitFile(self, ctx: semanticParser.FileContext):
        # print("File: %s" % ctx.getText())
        return self.visitSemantic(ctx.semantics())

    def visitSemantic(self, ctx: semanticParser.SemanticsContext):
        # print("Semantic: %s" % ctx.getText())
        for symbol in ctx.symbol():
            self.res.append(self.visitSymbol(symbol))
        return self.res

    def visitSymbol(self, ctx: semanticParser.SymbolContext):
        # print("Symbol: %s" % ctx.getText())
        symbol = {"type": self.visitType(ctx.type_())}
        if ctx.DASH() is not None:
            symbol["identifier"] = self.visitIdentifier(ctx.identifier())
        symbol["text"] = str(ctx.getText())
        return symbol

    def visitType(self, ctx: semanticParser.TypeContext):
        # print(ctx.getText())
        return str(ctx.getText())

    def visitIdentifier(self, ctx: semanticParser.IdentifierContext):
        return str(ctx.getText())


if __name__ == '__main__':
    with open("../../data/input/test/bona64_ref.txt") as f:
        res = parse(f.read())
        print(res)
