from semantic.sheet import EncodedSheet


def end2end_recognition(segments, model, vocabulary_path=None):
    """
    Utility function to perform the whole end-to-end OMR process on a list of segments.
    :param segments: list of segments
    :param model: to use on segments for predicting symbols. See for example ``CTC`` inside ``semantic/end2end/ctc_predict.py``
    :param vocabulary_path: path to the vocabulary file
    :return: an EncodedSheet (see ``semantic/sheet.py``) representing the whole image and a list of EncodedSheets
    for each of the segments provided
    """
    assert model is not None
    assert segments is not None
    sheet = EncodedSheet(vocabulary_path)
    subsheets = []

    for segment in segments:
        predictions = model.predict(segment)
        sheet.add_from_predictions(predictions)
        subsheet = EncodedSheet(vocabulary_path)
        subsheet.add_from_predictions(predictions)
        subsheets.append(subsheet)
    return sheet, subsheets




