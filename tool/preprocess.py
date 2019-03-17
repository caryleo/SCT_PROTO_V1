# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       preprocess.py
BY:             Gary 2019.3.14
LAST MODIFIED:  2019.3.16
DESCRIPTION:    preprocess core
"""

import json
import logging
import h5py
import numpy as np
import os
from PIL import Image
import skimage.io
import torch
from torchvision import transforms, models

import tool.netcore as netcore


def preprocess_captions(opts):
    """
    According to the opts to set the options about the preprocessing for captions
    Input json: the karpathy split
    Output h5: All the captions vocab-indexed (captions), the length for each caption (caption_lengths) and
        the start caption index (index_start) and end caption index (index_end) for each image
    Output json: An array for information of all images, including split (split), filepath (filepath,
        concatenating filename), coco image id (cocoid), and the width and height of the image if image
        directory specified (width, height). As well as an array for the vocabulary (index_to_word).
    :param opts: arguments
    :return: None
    """
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

    if image_root == "":
        logging.warning("No image root specified, width and height will not be stored")

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
    logging.info("Preprocess for captions complete")


def preprocess_features(opts, device):
    # load file path
    path_to_input_json = opts.input_caption_json
    directory_of_output = opts.output_feature_directory
    image_root = opts.image_root
    path_to_models = opts.model_directory

    assert path_to_input_json != "", "Path to input feature json is needed."
    assert directory_of_output != "", "Directory of output is needed."
    assert image_root != "", "Image Root is needed."
    assert path_to_models != "", "Path to models is needed."

    logging.info("Path to input feature json: %s" % path_to_input_json)
    logging.info("Directory of output: %s" % directory_of_output)
    logging.info("Image Root: %s" % image_root)
    logging.info("Path to models: %s" % path_to_models)

    # model for extraction
    model_name = opts.model
    attention_size = opts.attention_size

    logging.info("Model: %s" % model_name)
    logging.info("Attention Feature size: %d" % attention_size)

    # normalization
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # model, use the pretrained weights
    if model_name == "resnet50":
        model = models.resnet50()
    elif model_name == "resnet101":
        model = models.resnet101()
    else:
        model = models.resnet152()

    logging.info("Loading pretrained resnet model")
    model.load_state_dict(torch.load(os.path.join(path_to_models, model_name + ".pth")))
    logging.debug("Current Model: \n" + model.__str__())

    feature_net = netcore.my_resnet(model)
    feature_net.to(device=device)
    feature_net.eval()

    images = json.load(open(path_to_input_json, 'r'))["images"]
    num_images = len(images)

    # feature directories
    directory_of_fc_feature = os.path.join(directory_of_output, "features_fc")
    if not os.path.isdir(directory_of_fc_feature):
        os.mkdir(directory_of_fc_feature)
    logging.info("Directory of fc features: %s" % directory_of_fc_feature)
    directory_of_att_feature = os.path.join(directory_of_output, "features_att")
    if not os.path.isdir(directory_of_att_feature):
        os.mkdir(directory_of_att_feature)
    logging.info("Directory of att features: %s" % directory_of_att_feature)

    # feature extraction
    logging.info("Extracting features")
    for index, image in enumerate(images):
        input_image = skimage.io.imread(os.path.join(image_root, image["filepath"], image["filename"]))
        # gray_scale images
        if len(input_image.shape) == 2:
            input_image = input_image[:, :, np.newaxis]
            input_image = np.concatenate((input_image, input_image, input_image), axis=2)

        input_image = input_image.astype('float32') / 255.0
        input_img = torch.from_numpy(input_image.transpose([2, 0, 1])).to(device=device)
        input_img = normalize(input_img).to(device=device)
        with torch.no_grad():
            feat_fc, feat_att = feature_net(input_img, attention_size)

        np.save(os.path.join(directory_of_fc_feature, str(image['cocoid'])), feat_fc.data.cpu().float().numpy())
        np.savez_compressed(os.path.join(directory_of_att_feature, str(image['cocoid'])),
                            feat=feat_att.data.cpu().float().numpy())

        if index % 1000 == 0:
            logging.info('Processing %d / %d (%.2f%%)' % (index, num_images, index * 100.0 / num_images))

    logging.info("Extraction complete")
