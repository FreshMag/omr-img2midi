# Generated from /Users/fresh/Desktop/Universita/VisioneArtificiale/Progetto/omr-img2midi/data/semantic.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .semanticParser import semanticParser
else:
    from semanticParser import semanticParser

# This class defines a complete generic visitor for a parse tree produced by semanticParser.

class semanticVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by semanticParser#file.
    def visitFile(self, ctx:semanticParser.FileContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by semanticParser#semantics.
    def visitSemantics(self, ctx:semanticParser.SemanticsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by semanticParser#symbol.
    def visitSymbol(self, ctx:semanticParser.SymbolContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by semanticParser#type.
    def visitType(self, ctx:semanticParser.TypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by semanticParser#identifier.
    def visitIdentifier(self, ctx:semanticParser.IdentifierContext):
        return self.visitChildren(ctx)



del semanticParser