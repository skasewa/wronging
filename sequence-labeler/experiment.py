import sys
import collections
import numpy
import random
import math
import os
import gc

try:
    import ConfigParser as configparser
except:
    import configparser


from labeler import SequenceLabeler
from evaluator import SequenceLabelingEvaluator

def read_input_files(file_paths, max_sentence_length=-1, return_splits=False):
    """
    Reads input files in whitespace-separated format.
    Will split file_paths on comma, reading from multiple files.
    The format assumes the first column is the word, the last column is the label.
    If return_splits=True, also returns a vector of 'split_points' by signifying the sentence_id of the last sentence of each file.
    """
    sentences = []
    line_length = None
    split_points = []
    counter = -1    
    for file_path in file_paths.strip().split(","):
        with open(file_path, "r") as f:
            sentence = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    line_parts = line.split()
                    assert(len(line_parts) >= 2)
                    assert(len(line_parts) == line_length or line_length == None)
                    line_length = len(line_parts)
                    sentence.append(line_parts)
                elif len(line) == 0 and len(sentence) > 0:
                    if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                        sentences.append(sentence)
                        counter += 1
                    sentence = []
            if len(sentence) > 0:
                if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                    sentences.append(sentence)
                    counter += 1
        split_points += [counter]

    if return_splits:
        return sentences, split_points
    else:    
        return sentences



def parse_config(config_section, config_path):
    """
    Reads configuration from the file and returns a dictionary.
    Tries to guess the correct datatype for each of the config values.
    """
    config_parser = configparser.SafeConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    config = collections.OrderedDict()
    for key, value in config_parser.items(config_section):
        if value is None or len(value.strip()) == 0:
            config[key] = None
        elif value.lower() in ["true", "false"]:
            config[key] = config_parser.getboolean(config_section, key)
        elif value.isdigit():
            config[key] = config_parser.getint(config_section, key)
        elif is_float(value):
            config[key] = config_parser.getfloat(config_section, key)
        else:
            config[key] = config_parser.get(config_section, key)
    return config


def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def create_batches_of_sentence_ids(sentences, batch_equal_size, max_batch_size):
    """
    Groups together sentences into batches
    If batch_equal_size is True, make all sentences in a batch be equal length.
    If max_batch_size is positive, this value determines the maximum number of sentences in each batch.
    If max_batch_size has a negative value, the function dynamically creates the batches such that each batch contains abs(max_batch_size) words.
    Returns a list of lists with sentences ids.
    """
    batches_of_sentence_ids = []
    if batch_equal_size == True:
        sentence_ids_by_length = collections.OrderedDict()
        sentence_length_sum = 0.0
        for i in range(len(sentences)):
            length = len(sentences[i])
            if length not in sentence_ids_by_length:
                sentence_ids_by_length[length] = []
            sentence_ids_by_length[length].append(i)

        for sentence_length in sentence_ids_by_length:
            if max_batch_size > 0:
                batch_size = max_batch_size
            else:
                batch_size = int((-1.0 * max_batch_size) / sentence_length)

            for i in range(0, len(sentence_ids_by_length[sentence_length]), batch_size):
                batches_of_sentence_ids.append(sentence_ids_by_length[sentence_length][i:i + batch_size])
    else:
        current_batch = []
        max_sentence_length = 0
        for i in range(len(sentences)):
            current_batch.append(i)
            if len(sentences[i]) > max_sentence_length:
                max_sentence_length = len(sentences[i])
            if (max_batch_size > 0 and len(current_batch) >= max_batch_size) \
              or (max_batch_size <= 0 and len(current_batch)*max_sentence_length >= (-1 * max_batch_size)):
                batches_of_sentence_ids.append(current_batch)
                current_batch = []
                max_sentence_length = 0
        if len(current_batch) > 0:
            batches_of_sentence_ids.append(current_batch)
    return batches_of_sentence_ids



def process_sentences(data, labeler, is_training, learningrate, config, name):
    """
    Process all the sentences with the labeler, return evaluation metrics.
    """
    evaluator = SequenceLabelingEvaluator(config["main_label"], labeler.label2id, config["conll_eval"])
    batches_of_sentence_ids = create_batches_of_sentence_ids(data, config["batch_equal_size"], config["max_batch_size"])
    if is_training == True:
        random.shuffle(batches_of_sentence_ids)

    for sentence_ids_in_batch in batches_of_sentence_ids:
        batch = [data[i] for i in sentence_ids_in_batch]
        cost, predicted_labels, predicted_probs = labeler.process_batch(batch, is_training, learningrate)

        evaluator.append_data(cost, batch, predicted_labels)

        word_ids, char_ids, char_mask, label_ids = None, None, None, None
        while config["garbage_collection"] == True and gc.collect() > 0:
            pass

    results = evaluator.get_results(name)
    for key in results:
        print(key + ": " + str(results[key]))

    return results



def run_experiment(config_path):
    config = parse_config("config", config_path)
    temp_model_path = config_path + ".model"
    if "random_seed" in config:
        random.seed(config["random_seed"])
        numpy.random.seed(config["random_seed"])

    for key, val in config.items():
        print(str(key) + ": " + str(val))

    data_train, data_dev, data_test = None, None, None
    if config["path_train"] != None and len(config["path_train"]) > 0:
        if config['alternating_training']:
            # implements dataset-switching, i.e. first trains on the 'main' dataset, then on the augmented dataset in similar sized chunks
            data_train, split_points = read_input_files(config["path_train"], config["max_train_sent_length"], return_splits=True)
            main_train = data_train[:split_points[0]]
            data_train = data_train[split_points[0]:]
            random.shuffle(data_train)  # shuffle all augmented data
            data_train = main_train + data_train
            minibatch_size = split_points[0]
            minibatches = []
            for batch_start_index in range(0, len(data_train), minibatch_size):
                minibatches += [data_train[batch_start_index : batch_start_index+minibatch_size]]
            if len(minibatches[-1]) < 0.5 * minibatch_size:
                minibatches[-2] = minibatches[-2] + minibatches[-1]
                minibatches = minibatches[:-1]    # merge last minibatch with previous, if too small
        else:
            data_train = read_input_files(config["path_train"], config["max_train_sent_length"])
            minibatches = [data_train]
        print("minibatch sizes: "+", ".join([str(len(i)) for i in minibatches]))

    if config["path_dev"] != None and len(config["path_dev"]) > 0:
        data_dev = read_input_files(config["path_dev"])
    if config["path_test"] != None and len(config["path_test"]) > 0:
        data_test = []
        for path_test in config["path_test"].strip().split(":"):
            data_test += read_input_files(path_test)

    if config["load"] != None and len(config["load"]) > 0:
        labeler = SequenceLabeler.load(config["load"])
    else:
        labeler = SequenceLabeler(config)
        labeler.build_vocabs(data_train, data_dev, data_test, config["preload_vectors"])
        labeler.construct_network()
        labeler.initialize_session()
        if config["preload_vectors"] != None:
            labeler.preload_word_embeddings(config["preload_vectors"])

    print("parameter_count: " + str(labeler.get_parameter_count()))
    print("parameter_count_without_word_embeddings: " + str(labeler.get_parameter_count_without_word_embeddings()))

    if data_train != None:
        model_selector = config["model_selector"].split(":")[0]
        model_selector_type = config["model_selector"].split(":")[1]
        no_improvement_for = 0
        best_selector_value = 0.0
        best_epoch = -1
        learningrate = config["learningrate"]
        for epoch in range(config["epochs"]):
            print("EPOCH: " + str(epoch))
            for batchno, minibatch in enumerate(minibatches):
                print("BATCH: " + str(batchno))
                print("current_learningrate: " + str(learningrate))    
                random.shuffle(minibatch)
                results_train = process_sentences(minibatch, labeler, is_training=True, learningrate=learningrate, config=config, name="train")

                if data_dev != None:
                    results_dev = process_sentences(data_dev, labeler, is_training=False, learningrate=0.0, config=config, name="dev")
                    no_improvement_for += 1

                    if math.isnan(results_dev["dev_cost_sum"]) or math.isinf(results_dev["dev_cost_sum"]):
                        sys.stderr.write("ERROR: Cost is NaN or Inf. Exiting.\n")
                        break

                    if ((epoch == 0 and batchno == 0) or (model_selector_type == "high" and results_dev[model_selector] > best_selector_value) 
                                or (model_selector_type == "low" and results_dev[model_selector] < best_selector_value)):
                        best_epoch = epoch
                        best_batch = batchno
                        no_improvement_for = 0
                        best_selector_value = results_dev[model_selector]
                        labeler.saver.save(labeler.session, temp_model_path, latest_filename=os.path.basename(temp_model_path)+".checkpoint")
                    print("best_epoch and best_batch: " + str(best_epoch) + "-" + str(best_batch))
                    print("no improvement for: " + str(no_improvement_for))

                if no_improvement_for > config["learningrate_delay"]:
                    learningrate *= config["learningrate_decay"]

                if config["stop_if_no_improvement_for_epochs"] > 0 and no_improvement_for >= config["stop_if_no_improvement_for_epochs"]:
                    break

            if config["stop_if_no_improvement_for_epochs"] > 0 and no_improvement_for >= config["stop_if_no_improvement_for_epochs"]:
                break

            while config["garbage_collection"] == True and gc.collect() > 0:
                pass

        if data_dev != None and best_epoch >= 0:
            # loading the best model so far
            labeler.saver.restore(labeler.session, temp_model_path)

            os.remove(temp_model_path+".checkpoint")
            os.remove(temp_model_path+".data-00000-of-00001")
            os.remove(temp_model_path+".index")
            os.remove(temp_model_path+".meta")

    if config["save"] is not None and len(config["save"]) > 0:
        labeler.save(config["save"])

    if config["path_test"] is not None:
        i = 0
        for path_test in config["path_test"].strip().split(":"):
            data_test = read_input_files(path_test)
            results_test = process_sentences(data_test, labeler, is_training=False, learningrate=0.0, config=config, name="test"+str(i))
            i += 1


if __name__ == "__main__":
    run_experiment(sys.argv[1])

