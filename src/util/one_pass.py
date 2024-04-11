from semantic import CTC, EncodedSheet


def end2end_recognition(segments, model, vocabulary_path=None):
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


