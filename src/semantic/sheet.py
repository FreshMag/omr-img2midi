import os
import warnings

from semantic.parser import parse


class EncodedSheet:
    """
    Class used to maintain the produced semantic encoded in a simple format with a list of symbols. Used by other
    classes as output of the predictions of a model. It internally uses a list of symbol and a list of indexes. The
    symbols are just taken from the provided vocabulary at the correspondent indexes.
    """
    def __init__(self, vocabulary_file_path=None):
        """
        Constructor for the EncodedSheet class.
        :param vocabulary_file_path: path to the vocabulary file.
        """
        self.int2word = None
        self.output_symbols = []
        self.output_indexes = []
        self.set_vocabulary(vocabulary_file_path)

    def set_vocabulary(self, vocabulary_file_path):
        """
        Sets up the vocabulary.
        :param vocabulary_file_path: path to the vocabulary file.
        :return: None
        """
        if vocabulary_file_path is not None and os.path.exists(vocabulary_file_path):
            int2word = {}
            with open(vocabulary_file_path, 'r') as dict_file:
                for idx, word in enumerate(dict_file):
                    int2word[idx] = word.strip()

            self.int2word = int2word
            self.output_symbols = []
            if len(self.output_indexes) != 0:
                self.convert_indexes()
        else:
            warnings.warn("Vocabulary either not found or not provided. This sheet will work with indexes only until "
                          "you call 'set_vocabulary()' method correctly")

    def convert_indexes(self):
        """
        Converts the indexes into symbols. Useful if the vocabulary was not provided when the indexes were loaded.
        :return: None
        """
        if self.int2word is None or len(self.output_indexes) == 0:
            warnings.warn(
                "Warning: convert_indexes will not have any effect: neither indexes or vocabulary were found!")
            return
        else:
            for index in self.output_indexes:
                self.output_symbols.append(self.int2word[index])

    def add_from_semantic_file(self, semantic_file_path, file_format="index", separator="\t"):
        """
        Add the symbols to the sheet from the provided semantic file.
        :param semantic_file_path: path to the semantic file.
        :param file_format: format of the semantic file. Can be "index" or "symbol".
        :param separator: used in the file to separate symbols
        :return: None
        """
        with open(semantic_file_path, 'r') as semantic_file:
            predictions = []
            if file_format == "index":
                for value in semantic_file.read().split(separator):
                    try:
                        predictions.append(int(value))
                    except ValueError:
                        continue
                self.add_from_predictions(predictions)
            elif file_format == "symbol":
                if self.int2word is None:
                    raise ValueError("Vocabulary must be set in order to import symbols!")

                symbols = parse(semantic_file.read())
                for symbol in symbols:
                    matching_values = list(self.int2word.keys())[list(self.int2word.values()).index(symbol["text"])]
                    self.output_indexes.append(matching_values)
                    self.output_symbols.append(symbol)
            else:
                raise ValueError("file_format must be either 'index' or 'symbol'")

    def add_from_predictions(self, predictions, overwrite=False):
        """
        Like ``add_from_semantic_file()``, but adds the symbols to the sheet from the predictions of a model.
        :param predictions: output of the model
        :param overwrite: ``True`` to overwrite existing symbols
        :return: None
        """
        if overwrite:
            self.output_symbols = []
        for symbol_index in predictions:
            self.output_indexes.append(symbol_index)
            if self.int2word is not None:
                self.output_symbols.append(self.int2word[symbol_index])

    def print_symbols(self, separator=" "):
        """
        Utility function to print the symbols in the sheet.
        :param separator: used to separate each symbol
        :return: None
        """
        def print_list(to_print):
            for item in to_print:
                print(item, end=separator)
        if self.int2word is None:
            warnings.warn("Warning: vocabulary is not set! This will print only numerical indexes")
            print_list(self.output_indexes)
        else:
            print_list(self.output_symbols)

    def write_to_file(self, file_path, output_format="symbol", separator='\n'):
        """
        Outputs the content of this sheet to a file. It can be re-loaded later using the ``add_from_semantic_file``
        method.
        :param file_path: path to the output file.
        :param output_format: format used to output the symbols. Can be "symbol" or "index".
        :param separator: used to separate each symbol in the file
        :return: None
        """
        to_write = []
        if output_format == "symbol":
            to_write = self.output_symbols
        elif output_format == "index":
            to_write = self.output_indexes
        else:
            raise ValueError("Expected non empty output format")

        with open(file_path, 'w') as file:
            for value in to_write:
                file.write(str(value) + separator)
            file.write('\n')  # Adding a newline at the end
