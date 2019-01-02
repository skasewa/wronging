### Library to implement simple transforms required for data augmentation pipeline ###

import os, sys, random
from collections import defaultdict as dd
import numpy as np

class DataTSV(object):
    def __init__(self, main_label):
        self.data_list = []                     # list-of-list nested as data->[sent, label-string] pairs
        self.data_dict = dd(list)               # dict, of form sent:[label-string1,...]
        self.data_reverse_dict = dd(list)       # dict, of form num_errors:[[sent, label-string],...]
        self.main_label = main_label

    def strip_list(self):
        data_list = self.data_list
        data_list = [[sent_label_pair[0].strip(), sent_label_pair[1].strip()] for sent_label_pair in data_list]
        self.data_list = data_list

    def list_to_dict(self):
        self.strip_list()
        data_list = self.data_list
        data_dict = dd(list)
        for sent, label in data_list:
            if label not in data_dict[sent]:    # auto-remove duplicates from dictionary
                data_dict[sent] += [label]
        self.data_dict = data_dict

    def dict_to_reverse_dict(self):
        data_dict = self.data_dict
        main_label = self.main_label
        data_reverse_dict = dd(list)
        for sent in data_dict:
            for label in data_dict[sent]:
                num_errors = label.count(main_label)
                data_reverse_dict[num_errors] += [[sent, label]]
        self.data_reverse_dict = data_reverse_dict

    def reverse_dict_to_list(self):
        data_reverse_dict = self.data_reverse_dict
        data_list = []
        for num_errors in data_reverse_dict:
            data_list += data_reverse_dict[num_errors]
        self.data_list = data_list
        random.shuffle(self.data_list)

    def propagate_list(self):        
        self.list_to_dict()
        self.dict_to_reverse_dict()

    def propagate_dict(self):        
        self.dict_to_reverse_dict()
        self.reverse_dict_to_list()

    def propagate_rev_dict(self):
        self.reverse_dict_to_list()
        self.list_to_dict()

    def merge_tsv(self, tsv_data):
        self.data_list += tsv_data.data_list
        self.propagate_list()

    def shuffle_sents(self):
        random.shuffle(self.data_list)

    def remove_duplicates(self):
        self.propagate_dict()

    def remove_sents_outside_error_range(self, min_num_errors, max_num_errors):
        assert type(min_num_errors) == int, "min_num_errors should be type int! Aborting..."
        assert type(max_num_errors) == int, "max_num_errors should be type int! Aborting..."
        assert min_num_errors < max_num_errors, "min_num_errors should be less than max_num_errors! Aborting..."
        data_reverse_dict = self.data_reverse_dict
        num_error_list = list(data_reverse_dict.keys())
        for num_errors in num_error_list:
            if num_errors > max_num_errors or num_errors < min_num_errors:
                data_reverse_dict.pop(num_errors)
        self.data_reverse_dict = data_reverse_dict
        self.propagate_rev_dict()

def read_tsv(input_file, main_label='i'):
    # Read TSV file into DataTSV format
    tsv_data = DataTSV(main_label)
    data_list = []
    with open(input_file, "r") as f:
        sentence = []
        label = ''
        for line in f:
            line = line.strip()
            if len(line) > 0:
                line_parts = line.split()
                assert(len(line_parts) >= 2)
                sentence.append(line_parts[0])
                label += line_parts[-1]
            elif len(line) == 0 and len(sentence) > 0:
                data_list += [sentence, label]
                sentence = []
                label = ''
        if len(sentence) > 0:
            data_list += [sentence, label]
    tsv_data.data_list = data_list
    tsv_data.propagate_list()
    return tsv_data

def write_tsv(output_file, tsv_data, overwrite):
    # Write TSV from DataTSV object
    data = tsv_data.data_list
    if not overwrite:
        assert not os.path.exists(output_file), output_file + " already exists! Aborting..."
    with open(output_file, 'w') as f1:
        for [sent, label] in data:
            write_block = ""
            for i,token in enumerate(sent.split()):
                write_block += token + '\t' + label[i] + '\n'
            write_block += '\n'
            f1.write(write_block)

def read_txt(input_file):
    # Read text file, return list of sentences
    assert input_file[-4:] == '.txt', "Input file should be .txt! Aborting..."
    with open(input_file,'r') as f1:
        sents = f1.readlines()
    return sents

def write_txt(output_file, output_data, overwrite):
    # Write text file
    if not overwrite:
        assert not os.path.exists(output_file), output_file + " already exists! Aborting..."
    with open(output_file,'w') as f2:
        f2.write("".join(output_data))
        print("Wrote " + output_file)

def split_beam_search_txt(input_file, beams, keep=0, overwrite=False):
    # Split BS txt into many txts
    # returns list of filenames which have been written
    assert type(beams) == int, "param beams should be of type int! Aborting..."
    assert type(keep) == int, "param keep should be of type int! Aborting..."
    assert keep > -1, "param keep should be >= 0! Aborting..."

    if keep == 0: keep = beams
    
    file_pref = input_file[:-4]
    file_ext = '.txt'
    sents = read_txt(input_file)
    output_files = []
    for split_num in range(keep):
        output_file = file_pref + "-" + str(split_num) + file_ext
        output_data = sents[split_num::beams]  # slices data starting from 'split_num', in step size 'beams'
        write_txt(output_file, output_data, overwrite)
        output_files += [output_file]

    return output_files

max_sent_len = 500 + 1 #max number of tokens + 1
p_gap = 1.0
p_mismatch = 1.5 # will prefer one substitution over two gaps
p_static = np.zeros([max_sent_len,max_sent_len])
for i in range(max_sent_len):
    p_static[0,i] = i * p_gap
    p_static[i,0] = i * p_gap

def get_alignment(correct,wrong):
    # returns label-string corresponding to wrong sentences, e.g. 'ccicic'
    cor_mod = []
    wro_mod = []
    gap = "<gap>"
    
    c_tokens = correct.split()
    c_len = len(c_tokens) + 1  #using 1-indexing
    if c_len > max_sent_len:  #truncate if too long
        c_len = max_sent_len
        c_tokens = c_tokens[:max_sent_len-1]
    
    w_tokens = wrong.split()
    w_len = len(w_tokens) + 1 	#using 1-indexing
    if w_len > max_sent_len:  #truncate if too long
        w_len = max_sent_len
        w_tokens = w_tokens[:max_sent_len-1]

    penalties = np.array(p_static[0:c_len,0:w_len])  #make a copy of starting state with the correct size
    for i in range(1,c_len):
        for j in range(1,w_len):
            if c_tokens[i-1] == w_tokens[j-1]:
                p_ij = 0
            else:
                p_ij = p_mismatch
            penalties[i,j] = min(penalties[i-1,j-1] + p_ij,
                            penalties[i,j-1] + p_gap,
                            penalties[i-1,j] + p_gap)

    i = c_len-1
    j = w_len-1
    while i > 0 and j > 0:
        if penalties[i,j] == penalties[i,j-1] + p_gap:          # gap in correct sentence
            wro_mod += [w_tokens[j-1]]
            cor_mod += [gap]
            j = j-1
        elif penalties[i,j] == penalties[i-1,j] + p_gap:        # gap in incorrect sentence
            wro_mod += [gap]
            cor_mod += [c_tokens[i-1]]
            i = i-1
        else:                                                   #  match or substitution between them
            wro_mod += [w_tokens[j-1]]
            cor_mod += [c_tokens[i-1]]
            i = i-1
            j = j-1

    wro_mod.reverse()
    cor_mod.reverse()

    if i > 0:                                                   # put leading gaps for either on or the other
        wro_mod=[gap]*i+wro_mod
        cor_mod=c_tokens[0:i]+cor_mod
    elif j > 0:
        cor_mod=[gap]*j+cor_mod
        wro_mod=w_tokens[0:j]+wro_mod

    wro_mod.reverse()
    wro_mod_copy = wro_mod[:]
    for i,token in enumerate(wro_mod_copy):                     # remove trailing gaps from wrong_modified
        if token == gap:
            del wro_mod[0]
            continue
        else:
            break
    wro_mod.reverse()

    label_string = ''
    for i,wro_token in enumerate(wro_mod):
        cor_token = cor_mod[i]
        if wro_token == gap:
            continue
        if wro_token != cor_token:
            label_string += 'i'
        elif i > 0 and wro_mod[i-1] == gap:
            label_string += 'i'
        elif i == len(wro_mod) - 1 and len(cor_mod) > len(wro_mod):
            label_string += 'i'
        else:
            label_string += 'c'

    return label_string

def txt_to_error_detection_tsv(source_txt, target_txt, main_label='i'):
    # Convert txt to tsv
    # returns DataTSV data structure
    wrong_sents = read_txt(source_txt)
    correct_sents = read_txt(target_txt)
    assert len(wrong_sents) == len(correct_sents), "Files "+source_txt+" and "+target_txt+" do not have same number of sentences! Aborting..."
    data_list = []
    for wrong, correct in zip(wrong_sents, correct_sents):
        label_string = get_alignment(correct,wrong)
        data_list += [[wrong, label_string]]

    tsv_data = DataTSV(main_label)
    tsv_data.data_list = data_list
    tsv_data.propagate_list()
    return tsv_data

def txts_to_tsv_wrapper(correct_sents_file, corrupted_sents_files, output_tsv_file, min_num_errors, max_num_errors, duplicates, shuffle, overwrite):
    """Wraps all tasks to convert list of txts into tsv ; will work for AM and TS
    
    Arguments:
        correct_sents_file {str} -- filename of ground truth sentences
        corrupted_sents_files {list} -- list of filenames with corrupted sentences
        output_tsv_file {str} -- tsv filename to write out
        min_num_errors {int} -- sentences should have this minimum number of errors 
        max_num_errors {int} -- sentences should have this maximum number of errors 
        duplicates {boolean} -- keep duplicates? False discards them
        shuffle {boolean} -- shuffle sentences? False does not shuffle
        overwrite {boolean} -- overwrite existing files? False does not overwrite
    """
    
    main_label='i'
    data_tsv = DataTSV(main_label)
    for corrupt_file in corrupted_sents_files:
        print('aligning '+corrupt_file+' with '+correct_sents_file)
        beam_data_tsv_1 = txt_to_error_detection_tsv(corrupt_file, correct_sents_file, main_label)
        data_tsv.merge_tsv(beam_data_tsv_1)

    print('number of sentences:', len(data_tsv.data_list))

    print('removing duplicates...')
    data_tsv.remove_duplicates()
    print('number of sentences:', len(data_tsv.data_list))

    print('removing out-of-tolerance sentences...')
    data_tsv.remove_sents_outside_error_range(min_num_errors, max_num_errors)
    print('number of sentences:', len(data_tsv.data_list))

    print('shuffling sentences...')
    data_tsv.shuffle_sents()
    print('number of sentences:', len(data_tsv.data_list))

    print('writing tsv...')
    write_tsv(output_tsv_file, data_tsv, overwrite)


def beam_search_txt_to_tsv_wrapper(correct_sents_file, corrupted_sents_file, output_tsv_file, beams, keep_beams, min_num_errors, max_num_errors, duplicates, shuffle, overwrite):
    """Wraps all tasks to convert beam-search output into tsv 
    
    Arguments:
        correct_sents_file {str} -- filename of ground truth sentences
        corrupted_sents_file {str} -- filename of corrupted sentences by beam search
        output_tsv_file {str} -- tsv filename to write out
        beams {int} -- number of beams used during beam search
        keep_beams {int} -- number of top beams to keep 
        min_num_errors {int} -- sentences should have this minimum number of errors 
        max_num_errors {int} -- sentences should have this maximum number of errors 
        duplicates {boolean} -- keep duplicates? False discards them
        shuffle {boolean} -- shuffle sentences? False does not shuffle
        overwrite {boolean} -- overwrite existing files? False does not overwrite
    """
    
    print('splitting beam-search file into aligned files...')
    beam_txt_filenames = split_beam_search_txt(corrupted_sents_file, beams, keep_beams, overwrite)
    txts_to_tsv_wrapper(correct_sents_file, beam_txt_filenames,output_tsv_file, min_num_errors, max_num_errors, duplicates, shuffle, overwrite)

def beam_search_wrapper_sample():
    correct_sents_file = '/home/user/wronging/data/train/fce/targets.txt'
    corrupted_sents_file = '/home/user/wronging/data/NMT/fce-bs-11-50000.txt'
    output_tsv_file = '/home/user/wronging/data/NMT/fce-bs-3.tsv'
    beams = 11 
    keep_beams = 3
    min_num_errors = 0 
    max_num_errors = 5
    duplicates = False
    shuffle = True
    overwrite = True

    beam_search_txt_to_tsv_wrapper(correct_sents_file, corrupted_sents_file, output_tsv_file, beams, keep_beams, min_num_errors, max_num_errors, duplicates, shuffle, overwrite)

def wrapper_sample():
    correct_sents_file = '/home/user/wronging/data/train/fce/targets.txt'
    corrupted_sents_file = ['/home/user/wronging/data/NMT/fce-am-50000.txt', '/home/user/wronging/data/NMT/fce-am-51000.txt', '/home/user/wronging/data/NMT/fce-am-52000.txt']
    output_tsv_file = '/home/user/wronging/data/NMT/fce-am-3.tsv'
    min_num_errors = 0 
    max_num_errors = 5
    duplicates = False
    shuffle = True
    overwrite = True

    txts_to_tsv_wrapper(correct_sents_file, corrupted_sents_file, output_tsv_file, min_num_errors, max_num_errors, duplicates, shuffle, overwrite)

if __name__ == "__main__":
    wrapper_sample()