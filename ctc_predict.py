"""

    Originally authored by Jorge Calvo Zaragoza <https://github.com/calvozaragoza>

    Modified by Francesco Magnani <https://github.com/FreshMag>

    Licensed under the MIT License (see LICENSE for details)

"""

import argparse
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np

from midi_translator import MidiTranslator, MidiPlayer
from utils import EncodedSheet

"""
Disables eager execution: introduced after script conversion using tf_upgrade_v2 script 
https://www.tensorflow.org/guide/migrate/upgrade?hl=en
"""
tf.compat.v1.disable_eager_execution()


class CTC:
    def __init__(self, model_file_path="./models/semantic/semantic_model.meta"):
        tf.compat.v1.reset_default_graph()
        self.session = tf.compat.v1.InteractiveSession()

        # Restore weights
        saver = tf.compat.v1.train.import_meta_graph(model_file_path)
        saver.restore(self.session, model_file_path[:-5])

        graph = tf.compat.v1.get_default_graph()

        self.input_model = graph.get_tensor_by_name("model_input:0")
        self.seq_len = graph.get_tensor_by_name("seq_lengths:0")
        self.rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
        height_tensor = graph.get_tensor_by_name("input_height:0")
        width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
        logits = tf.compat.v1.get_collection("logits")[0]

        # Constants that are saved inside the model itself
        self.width_reduction, self.height = self.session.run([width_reduction_tensor, height_tensor])

        self.decoded, _ = tf.nn.ctc_greedy_decoder(logits, self.seq_len)

    def predict(self, image_file_path):
        image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
        image = ctc_utils.resize(image, self.height)
        image = ctc_utils.normalize(image)
        image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)

        seq_lengths = [image.shape[2] / self.width_reduction]

        prediction = self.session.run(self.decoded,
                                      feed_dict={
                                          self.input_model: image,
                                          self.seq_len: seq_lengths,
                                          self.rnn_keep_prob: 1.0,
                                      })

        str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)

        return str_predictions[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
    parser.add_argument('-image', dest='image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
    args = parser.parse_args()

    sheet = EncodedSheet(args.voc_file)
    model = CTC(args.model)
    print("Processing image...")
    sheet.add_from_predictions(model.predict(args.image))
    print("Done!")
    print("Symbols found:")
    sheet.print_symbols()

    print("Converting to MIDI...")
    output_midi_path = "output.mid"
    translator = MidiTranslator(sheet)
    translator.translate(output_midi_path)
    print("Done!")

    print("Converting to audio...")
    player = MidiPlayer(output_midi_path)
    player.to_audio_file()
    print("Done!")
