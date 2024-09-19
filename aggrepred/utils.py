import torch


# Amino acids in a specific order as per original Parapred implementation
aa = "CSTPAGNDEQHRKMILVFYW-"
idx2aa = dict([(i, v) for i, v in enumerate(aa)])
aa2idx = dict([(v, i) for i, v in enumerate(aa)])
NUM_AMINOS = len(aa)

# Meiler features
MEILER = {
    "C": [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    "S": [1.31, 0.06, 1.6, -0.04, 5.7, 0.2, 0.28],
    "T": [3.03, 0.11, 2.6, 0.26, 5.6, 0.21, 0.36],
    "P": [2.67, 0.0, 2.72, 0.72, 6.8, 0.13, 0.34],
    "A": [1.28, 0.05, 1.0, 0.31, 6.11, 0.42, 0.23],
    "G": [0.0, 0.0, 0.0, 0.0, 6.07, 0.13, 0.15],
    "N": [1.6, 0.13, 2.95, -0.6, 6.52, 0.21, 0.22],
    "D": [1.6, 0.11, 2.78, -0.77, 2.95, 0.25, 0.2],
    "E": [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
    "Q": [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
    "H": [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.3],
    "R": [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
    "K": [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    "M": [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
    "I": [4.19, 0.19, 4.0, 1.8, 6.04, 0.3, 0.45],
    "L": [2.59, 0.19, 4.0, 1.7, 6.04, 0.39, 0.31],
    "V": [3.67, 0.14, 3.0, 1.22, 6.02, 0.27, 0.49],
    "F": [2.94, 0.29, 5.89, 1.79, 5.67, 0.3, 0.38],
    "Y": [2.94, 0.3, 6.47, 0.96, 5.66, 0.25, 0.41],
    "W": [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
    "X": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# Convert Meiler features to PyTorch Tensors
MEILER = dict([(k, torch.Tensor(v)) for k, v in MEILER.items()])
NUM_MEILER = 7



###############################################
# ENCODING
###############################################

from aggrepred.utils import *
from typing import List



NUM_FEATURES = NUM_AMINOS + NUM_MEILER
def onehot_encode(sequence: str, max_length: int = 1000) -> torch.Tensor:
    """
    One-hot encode an amino acid sequence

    :param sequence:   protein sequence
    :param max_length: specify the maximum length for protein sequence to use, it helps to have have size in batches

    :return: max_length x num_features tensor
    """
    
    seqlen = len(sequence)
    use_length = max_length
    # use_length = min(seqlen,max_length) #variable to len of seq, if want fix_size like 1024, set it to fix
    encoded = torch.zeros((use_length, NUM_AMINOS))
    for i in range(min(seqlen, max_length)):
        aa = sequence[i]
        encoded[i][aa2idx.get(aa, NUM_AMINOS-1)] = 1
    return encoded

def onehot_encode_batch(sequences: list, max_length: int = 1000) -> torch.Tensor:
    """
    One-hot encode an amino acid sequence

    :param sequence:   protein sequence
    :param max_length: specify the maximum length for protein sequence to use, it helps to have have size in batches

    :return: max_length x num_features tensor
    """
    batch_size = len(sequences)
    batch_encoded = torch.zeros((batch_size, max_length, NUM_AMINOS))
    for i, seq in enumerate(sequences):
        batch_encoded[i] = onehot_encode(seq, max_length)
    return batch_encoded

def onehot_meiler_encode(sequence: str, max_length: int = 1000) -> torch.Tensor:
    """
    One-hot encode an amino acid sequence, then concatenate with Meiler features.

    :param sequence:   protein sequence
    :param max_length: specify the maximum length for protein sequence to use, it helps to have have size in batches

    :return: max_length x num_features tensor
    """
    
    seqlen = len(sequence)
    use_length = max_length
    # use_length = min(seqlen,max_length) #variable to len of seq, if want fix_size like 1024, set it to fix
    encoded = torch.zeros((use_length, NUM_AMINOS+ NUM_MEILER))
    for i in range(min(seqlen, max_length)):
        aa = sequence[i]
        encoded[i][aa2idx.get(aa, NUM_AMINOS-1)] = 1
        encoded[i][-NUM_MEILER:] = MEILER[aa] if aa in MEILER else MEILER["X"]
    return encoded


def onehot_meiler_encode_batch(sequences: list, max_length: int = 1000) -> torch.Tensor:
    """
    One-hot encode an amino acid sequence, then concatenate with Meiler features.

    :param sequence:   protein sequence
    :param max_length: specify the maximum length for protein sequence to use, it helps to have have size in batches

    :return: max_length x num_features tensor
    """
    batch_size = len(sequences)
    batch_encoded = torch.zeros((batch_size, max_length, NUM_AMINOS+ NUM_MEILER))
    for i, seq in enumerate(sequences):
        batch_encoded[i] = onehot_meiler_encode(seq, max_length)
    return batch_encoded

def embed_esm_batch(batch_sequences, model, alphabet, repr_layer='last'):
    batch_converter = alphabet.get_batch_converter()
    data = [("protein" + str(i), seq) for i, seq in enumerate(batch_sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)

    # Get the embeddings from the last layer
    last_layer = model.num_layers
    token_embeddings = results["representations"][last_layer]
    
    return token_embeddings[:,1:-1,:]

def embed_protbert_batch(sequences, model, tokenizer, device='cuda' ):
    model.eval()

    sequences_w_spaces = [' '.join(list(seq)) for seq in sequences]
    processed_sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_w_spaces]

    ids = tokenizer.batch_encode_plus(processed_sequences, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]

    return embedding[:,1:-1,:]


###############################################



import pandas as pd
import re

def format_pdb(pdb_file):
    '''
    Process pdb file into pandas df
    
    Original author: Alissa Hummer
    
    :param pdb_file: file path of .pdb file to convert
    :returns: df with atomic level info
    '''
    
    pd.options.mode.chained_assignment = None
    pdb_whole = pd.read_csv(pdb_file,header=None,delimiter='\t')
    pdb_whole.columns = ['pdb']
    pdb = pdb_whole[pdb_whole['pdb'].str.startswith('ATOM')]
    pdb['Atom_Num'] = pdb['pdb'].str[6:11].copy()
    pdb['Atom_Name'] = pdb['pdb'].str[11:16].copy()
    pdb['AA'] = pdb['pdb'].str[17:20].copy()
    pdb['Chain'] = pdb['pdb'].str[20:22].copy()
    pdb['Res_Num'] = pdb['pdb'].str[22:27].copy().str.strip()
    pdb['x'] = pdb['pdb'].str[27:38].copy()
    pdb['y'] = pdb['pdb'].str[38:46].copy()
    pdb['z'] = pdb['pdb'].str[46:54].copy()#
    pdb['bfactor'] = pdb['pdb'].str[60:66].copy()#
    pdb['Atom_type'] = pdb['pdb'].str[77].copy()
    pdb.drop('pdb',axis=1,inplace=True)
    pdb.replace({' ':''}, regex=True, inplace=True)
    pdb.reset_index(inplace=True)
    pdb.drop('index',axis=1,inplace=True)
    
    # remove H atoms from our data (interested in heavy atoms only)
    pdb = pdb[pdb['Atom_type']!='H']

    return pdb


def get_ordered_AA_3_letter_codes():
    '''
    '''
    AA_unique_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                       'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                       'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                       'SER', 'THR', 'TRP', 'TYR', 'VAL']
    return AA_unique_names


def get_normal_CDR_loop_start_and_end_vals(loop_num):
    '''
    These are the normal IMGT start and end res nums for each CDR loop
    
    :param loop_num: int 1, 2, or 3
    :return: two-element list containing str of start and end nums
    '''
    if loop_num == 1:
        start, end = "27", "38"
    elif loop_num == 2:
        start, end = "56", "65"
    elif loop_num == 3:
        start, end = "105", "117"
    else:
        raise ValueError("loop_num must be either integer 1, 2, or 3")
    return start, end


def get_normal_CDRplus2_loop_start_and_end_vals(loop_num):
    '''
    These are the normal IMGT start and end res nums for each CDR loop + 2 extra res
    
    :param loop_num: int 1, 2, or 3
    :return: two-element list containing str of start and end nums
    '''
    if loop_num == 1:
        start, end = "25", "40"
    elif loop_num == 2:
        start, end = "54", "67"
    elif loop_num == 3:
        start, end = "103", "119"
    else:
        raise ValueError("loop_num must be either integer 1, 2, or 3")
    return start, end


def get_normal_Fv_start_and_end_vals(heavy=False):
    '''
    These are the normal IMGT start and end res nums for the Fv region
    
    :return: two-element list containing str of start and end nums
    '''
    start = "1"
    end = "128" if heavy else "127"
    return start, end


def search_up_for_nearest(start, end, res_num_list):
    '''
    In case start num isn't found, search for nearest res that is still in loop
    
    :param start: str res num that normally indicates start of loop
    :param end: str res num that normally indicates end of loop
    :param res_num_list: ordered list of strs of all res nums in one chain from pdb
    :return: nearest res to original start of loop or None if not found
    '''
    start = int(start)
    end = int(end)
    new_start = None

    for res_num in range(start, end+1):
        if str(res_num) in res_num_list:
            new_start = str(res_num)
            break
        res_num_flexi_insertion = re.compile(str(res_num) + "[A-Z]")
        tmp_list = list(filter(res_num_flexi_insertion.match, res_num_list))
        try:
            new_start = tmp_list[0]
            break
        except IndexError:
            pass
        
    return new_start


def search_down_for_nearest(start, end, res_num_list):
    '''
    In case end num isn't found, search for nearest res that is still in loop
    
    :param start: str res num that normally indicates start of loop
    :param end: str res num that normally indicates end of loop
    :param res_num_list: ordered list of strs of all res nums in one chain from pdb
    :return: nearest res to original end of loop or None if not found
    '''
    rev_res_num_list = res_num_list.copy()
    rev_res_num_list.reverse()
    start = int(start)
    end = int(end)
    new_end = None
    
    for res_num in range(end, start-1, -1):
        if str(res_num) in rev_res_num_list:
            new_end = str(res_num)
            break
        res_num_flexi_insertion = re.compile(str(res_num)+"[A-Z]")
        tmp_list = list(filter(res_num_flexi_insertion.match, rev_res_num_list))
        try:
            new_end = tmp_list[0]
            break
        except IndexError:
            pass
        
    return new_end


###############################################


# import os
# import pickle
# import torch
# import math
# import sys
# import csv
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# from torch.autograd import Variable
# from sklearn.utils import compute_class_weight
# from torch.utils.data.sampler import SubsetRandomSampler

# #############################################
# # Utils
# #############################################


# def sort_batch(lengths, others):
#     """
#     Sort batch data and labels by length
#     Args:
#         lengths (nn.Tensor): tensor containing the lengths for the data

#     Returns:

#     """
#     batch_size = lengths.size(0)

#     sorted_lengths, sorted_idx = lengths.sort()
#     reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()
#     sorted_lengths = sorted_lengths[reverse_idx]

#     return sorted_lengths, (lst[sorted_idx][reverse_idx] for lst in others)


# def progress(loss, epoch, batch, batch_size, dataset_size):
#     batches = math.ceil(float(dataset_size) / batch_size)
#     count = batch * batch_size
#     bar_len = 20
#     filled_len = int(round(bar_len * count / float(dataset_size)))

#     bar = '=' * filled_len + '-' * (bar_len - filled_len)

#     status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
#     _progress_str = "\r \r [{}] ...{}".format(bar, status)
#     sys.stdout.write(_progress_str)
#     sys.stdout.flush()

#     if batch == batches:
#         print()


# def get_class_weights(y):
#     """
#     Returns the normalized weights for each class
#     based on the frequencies of the samples
#     :param y: list of true labels (the labels must be hashable)
#     :return: dictionary with the weight for each class
#     """

#     weights = compute_class_weight('balanced', np.unique(y), y)

#     d = {c: w for c, w in zip(np.unique(y), weights)}

#     return d


# def class_weigths(targets):
#     w = get_class_weights(targets)
#     labels = get_class_labels(targets)
#     return torch.FloatTensor([w[l] for l in sorted(labels)])


# def get_class_labels(y):
#     """
#     Get the class labels
#     :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
#     :return: sorted unique class labels
#     """
#     return np.unique(y)


# def split_train_set(train_set, contiguous, split_rate=0.1):

#     """
#     Get the class labels
#     :param train_set: list training indexes
#     :param contiguous: contiguous or non-contiguous split
#     :param split_rate: split rate (default: 0.1)
#     :return: train and validation samplers for torch.utils.data.DataLoader
#     """

#     train_len = len(train_set.data)
#     indices = list(range(train_len))
#     split = int(train_len * split_rate)

#     if contiguous:
#         train_idx, validation_idx = indices[split:], indices[:split]
#     else:
#         validation_idx = np.random.choice(indices, size=split, replace=False)
#         train_idx = list(set(indices) - set(validation_idx))

#     train_sampler = SubsetRandomSampler(train_idx)
#     validation_sampler = SubsetRandomSampler(validation_idx)

#     return train_sampler, validation_sampler



# def get_labels_to_categories_map(y):
#     """
#     Get the mapping of class labels to numerical categories
#     :param y: list of labels, ex. ['positive', 'negative', 'positive', 'neutral', 'positive', ...]
#     :return: dictionary with the mapping
#     """
#     labels = get_class_labels(y)
#     return {l: i for i, l in enumerate(labels)}


# def save_model(model, file):

#     torch.save(model, file)


# def df_to_csv(df,file):
#     df.to_csv(file, sep=',', index=True)


# def csv_to_df(file):
#     df = pd.read_csv(file, sep=',')
#     return df


# def write_to_csv(file, data):
#     with open(file_cache_name(file), 'wb') as csv_file:
#         spamwriter = csv.writer(csv_file, delimiter=' ')
#         for line in data:
#             spamwriter.writerow(line)


# def read_from_csv(file, data):
#     with open(file_cache_name(file), 'wb') as csv_file:
#         spamwriter = csv.reader(csv_file, delimiter=' ')
#         for line in data:
#             data.add(line)


# def file_cache_name(file):
#     head, tail = os.path.split(file)
#     filename, ext = os.path.splitext(tail)
#     return os.path.join(head, filename + ".p")


# def write_cache_word_vectors(file, data):
#     with open(file_cache_name(file), 'wb') as pickle_file:
#         pickle.dump(data, pickle_file)


# def load_cache_word_vectors(file):
#     with open(file_cache_name(file), 'rb') as f:
#         return pickle.load(f)

# #############################################
# # Ploters
# #############################################


# def loss_curve(df,EPOCHS):
#     plt.figure()
#     plt.plot(range(1, EPOCHS + 1), df["Train_Loss"], 'b', label='training loss')
#     plt.plot(range(1, EPOCHS + 1), df["Val_Loss"], 'r', label='validation loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.title('Train/Val Loss')
#     plt.legend(loc='best')
#     plt.show()


# def f1_curve(df,EPOCHS):
#     plt.figure()
#     plt.plot(range(1, EPOCHS + 1), df["Train_F1"], 'b', label='training F1')
#     plt.plot(range(1, EPOCHS + 1), df["Val_F1"], 'r', label='validation F1')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.title('Train/Val F1')
#     plt.legend(loc='best')
#     plt.grid()
#     plt.show()


# def acc_curve(df,EPOCHS):
#     plt.figure()
#     plt.plot(range(1, EPOCHS + 1), df["Train_Acc"], 'b', label='training acc')
#     plt.plot(range(1, EPOCHS + 1), df["Val_Acc"], 'r', label='validation acc')
#     plt.ylabel('acc')
#     plt.xlabel('epoch')
#     plt.title('Train/Val Acc')
#     plt.legend(loc='best')
#     plt.grid()
#     plt.show()


# def mae_curve(df, EPOCHS):
#     plt.figure()
#     plt.plot(range(1, EPOCHS + 1), df["Macro_MAE"], 'r', label='macro mean ab error')
#     plt.plot(range(1, EPOCHS + 1), df["Micro_MAE"], 'b', label='micro mean ab error')
#     plt.ylabel('mae')
#     plt.xlabel('epoch')
#     plt.title('Mean Absolute Error')
#     plt.legend(loc='best')
#     plt.grid()
#     plt.show()

# #############################################
# # Model Evaluator
# #############################################


# def eval_dataset(dataloader, model, loss_function):
#     # switch to eval mode -> disable regularization layers, such as Dropout
#     model.eval()

#     y_pred = []
#     y = []

#     total_loss = 0
#     for i_batch, sample_batched in enumerate(dataloader, 1):
#         # get the inputs (batch)
#         # inputs, topics, labels, lengths, indices = sample_batched
#         inputs, topics, labels, lengths, topic_lengths, weights, indices= sample_batched

#         # sort batch (for handling inputs of variable length)
#         lengths, (inputs, labels, topics) = sort_batch(lengths, (inputs, labels, topics))

#         # convert to CUDA Variables
#         if torch.cuda.is_available():
#             inputs = Variable(inputs.cuda())
#             topics = Variable(topics.cuda())
#             labels = Variable(labels.cuda())
#             lengths = Variable(lengths.cuda())
#             topic_lengths = Variable(topic_lengths.cuda())
#         else:
#             inputs = Variable(inputs)
#             topics = Variable(topics)
#             labels = Variable(labels)
#             topic_lengths = Variable(topic_lengths)

#         outputs = model(inputs, topics, lengths, topic_lengths)

#         loss = loss_function(outputs, labels)
#         total_loss += loss.data[0]

#         _, predicted = torch.max(outputs.data, 1)

#         y.extend(list(labels.data.cpu().numpy().squeeze()))
#         y_pred.extend(list(predicted.squeeze()))

#     avg_loss = total_loss / i_batch

#     return avg_loss, (y, y_pred)