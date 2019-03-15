# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       preprocess.py
BY:             Gary 2019.3.14
LAST MODIFIED:  2019.3.15
DESCRIPTION:    preprocess core
"""
import json
import logging
import h5py
import numpy as np
import os
from PIL import Image


def preprocess_captions(opts):
    # load file path
    path_to_input_json = opts.input_caption_json
    path_to_output_json = opts.output_caption_json
    path_to_output_h5 = opts.output_caption_h5
    image_root = opts.image_root

    assert path_to_input_json != "", "Path to input caption json is needed."
    assert path_to_output_json != "", "Path to output caption json is needed."
    assert path_to_output_h5 != "", "Path to output caption h5 is needed."

    logging.info("Path to input caption json: %s" % path_to_input_json)
    logging.info("Path to output caption json: %s" % path_to_output_json)
    logging.info("Path to output caption h5: %s" % path_to_output_h5)
    logging.info("Image root: %s" % image_root)

    # load images and dataset
    inputs = json.load(open(path_to_input_json, 'r'))
    images = inputs["images"]
    num_images = len(images)
    num_captions = sum(len(image["sentences"]) for image in images)
    dataset = inputs["dataset"]
    logging.debug("Processing dataset: %s" % dataset)
    logging.debug("Number of images recorded: %d" % num_images)
    logging.debug("Number of captions recorded: %d" % num_captions)

    # load word threshold and maximal sentence length
    word_threshold = opts.word_threshold
    max_sentence_length = opts.max_sentence_length
    logging.info("Word occurrences threshold: %d" % word_threshold)
    logging.info("Maximal sentence length: %d" % max_sentence_length)

    # count word occurrences and sentence length
    occurrences = dict()
    lengths = dict()
    for image in images:
        for sentence in image["sentences"]:
            length = len(sentence["tokens"])
            lengths[length] = lengths.get(length, 0) + 1
            for word in sentence["tokens"]:
                occurrences[word] = occurrences.get(word, 0) + 1

    # DEBUG: sort it! big first!
    ordered_occurrences = sorted([(times, word) for word, times in occurrences.items()], reverse=True)
    logging.debug("Top 10 common words:\n" + "\n".join(map(str, ordered_occurrences[:10])))

    # statistics about occurrences
    sum_words = sum(occurrences.values())
    vocabulary = list()
    rare_words = list()
    for word, times in occurrences.items():
        if times <= word_threshold:
            rare_words.append(word)
        else:
            vocabulary.append(word)

    sum_rare_words = sum(occurrences[word] for word in rare_words)
    logging.info("Size of vocabulary: %d" % (len(vocabulary)))
    logging.info("Number of rare words: %d / %d (%.2f%%)" %
                 (len(rare_words), len(occurrences), len(rare_words) * 100.0 / len(occurrences)))
    logging.info("Number of UNK replacements: %d / %d (%.2f%%)" %
                 (sum_rare_words, sum_words, sum_rare_words * 100.0 / sum_words))

    # statistics about sentences length
    max_length = max(lengths.keys())
    sum_sentences = sum(lengths.values())
    logging.info("Maximal sentence length: %d" % max_length)
    logging.debug("Distribution of sentence lengths (length | number | ratio):")
    for i in range(max_length + 1):
        logging.debug("%2d | %7d | %2.5f%%" % (i, lengths.get(i, 0), lengths.get(i, 0) * 100.0 / sum_sentences))

    # insect the token UNK
    if sum_rare_words > 0:
        logging.info("Inserting the token UNK")
        vocabulary.append("UNK")

    # create mapping between index and word, 1-index
    index_to_word = dict()
    word_to_index = dict()
    for index, word in enumerate(vocabulary, start=1):
        index_to_word[index] = word
        word_to_index[word] = index

    # encode all captions into a large array for h5 storage, 1-indexed
    logging.info("Encoding all captions into one array")
    array_captions = list()
    array_index_start = np.zeros(num_images, dtype='uint32')
    array_index_end = np.zeros(num_images, dtype='uint32')
    array_lengths = np.zeros(num_captions, dtype='uint32')

    # 0-indexed
    bcount = 0
    count = 1
    for index, image in enumerate(images):
        num = len(image["sentences"])
        assert num > 0, "No caption for this image???"

        captions = np.zeros((num, max_sentence_length), dtype='uint32')
        for tag, sentence in enumerate(image["sentences"]):
            array_lengths[bcount] = min(max_sentence_length, len(sentence))
            bcount += 1
            for pos, word in enumerate(sentence["tokens"]):
                if pos < max_sentence_length:
                    captions[tag, pos] = word_to_index[word] if occurrences[word] > word_threshold else word_to_index[
                        "UNK"]

        array_captions.append(captions)
        array_index_start[index] = count
        array_index_end[index] = count + num - 1
        count += num

    # concatenate together
    all_captions = np.concatenate(array_captions, axis=0)
    logging.debug("Size of the captions array: " + str(all_captions.shape))
    assert all_captions.shape[0] == num_captions, "Numbers are not matched, something is wrong???"
    assert np.all(array_lengths > 0), "Some captions have no words???"
    logging.info("Encode all captions into one array complete")

    # create the h5 file
    logging.info("Creating h5 file: %s" % path_to_output_h5)
    output_h5 = h5py.File(path_to_output_h5, 'w')
    output_h5.create_dataset("captions", dtype='uint32', data=all_captions)
    output_h5.create_dataset("index_start", dtype='uint32', data=array_index_start)
    output_h5.create_dataset("index_end", dtype='uint32', data=array_index_end)
    output_h5.create_dataset("caption_lengths", dtype='uint32', data=array_lengths)
    output_h5.close()
    logging.info("Create h5 file complete")

    # create the json file
    logging.info("Creating json file: %s" % path_to_output_json)
    output_json = dict()
    output_json["index_to_word"] = index_to_word
    output_json["images"] = list()

    for index, image in enumerate(images):
        output_image = dict()
        output_image["split"] = image["split"]
        output_image["filepath"] = os.path.join(image["filepath"], image["filename"])
        output_image["cocoid"] = image["cocoid"]
        if image_root != "":
            with Image.open(os.path.join(image_root, output_image["filepath"])) as img:
                output_image["width"], output_image["height"] = img.size

        output_json["images"].append(output_image)

    json.dump(output_json, open(path_to_output_json, 'w'))
    logging.info("Create json file complete")
