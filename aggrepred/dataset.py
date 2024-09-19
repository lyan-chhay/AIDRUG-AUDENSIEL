import torch
import pandas as pd
import ast
import sys
import os
import re
from transformers import BertTokenizer, BertModel

top_folder_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, top_folder_path)

from aggrepred.utils import *



class Dataset:
    def __init__(self, csv_file, max_seq_len=1024, encode_type='onehot'):
        self.data = pd.read_csv(csv_file)
        self.data['scores'] = self.data['scores'].apply(ast.literal_eval)

        def count_pos_neg_values(lst):
            count_pos = sum(1 for x in lst if x > 0)
            count_neg = sum(1 for x in lst if x <= 0)
            return count_pos, count_neg

        # # Apply the function to create new columns
        # self.data[['count_positive', 'count_negative']] = self.data['scores'].apply(count_pos_neg_values).apply(pd.Series)
        # self.data['len'] = self.data['scores'].apply(lambda x: len(x))

        # self.data['neg_to_pos_ratio'] = self.data['count_negative'] / self.data['count_positive']

        self.max_seq_len = max_seq_len


        # self.encode_type= encode_type

        # # Pre-encode sequences if using ProtBERT
        # if self.encode_type == 'protbert':
        #     print("downlaod ProtBert tokenizer and model:")
        #     self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False,resume_download=None)
        #     self.model = BertModel.from_pretrained("Rostlab/prot_bert",resume_download=None)
        #     self.pre_encode_sequences()

    
    # def pre_encode_sequences(self):
    #     # Ensure the tokenizer and model are only downloaded once
    #     # tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    #     # model = BertModel.from_pretrained("Rostlab/prot_bert")

    #     sequences = self.data['sequence'].tolist()
    #     print(sequences[:3])
    #     sequences = [" ".join(seq) for seq in sequences]
    #     sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]

    #     print(sequences[:3])
    #     encoded_inputs = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=self.max_seq_len)
        
    #     with torch.no_grad():
    #         outputs = self.model(**encoded_inputs)
        
    #     self.encoded_seqs = outputs["last_hidden_state"]

    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # def prepare_protbert(sequence):
        #     sequence = " ".join(sequence)
        #     sequence = re.sub(r"[UZOB]", "X", sequence) 
        #     tokenized_input = self.tokenizer(sequence, 
        #                                         max_length = self.max_seq_len,
        #                                         padding='max_length',
        #                                         truncation=True,
        #                                         return_attention_mask=True,
        #                                         return_tensors='pt', )
        #     # print(tokenized_input)
        #     out = self.model(**tokenized_input)
        #     return out['last_hidden_state'].squeeze(0)
        #     # return tokenized_input
    
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        
        row = self.data.iloc[idx]
        seq = row['sequence']
        scores = row['scores']

        y  = scores[:self.max_seq_len] + [0] * (self.max_seq_len - len(scores))
        # y = torch.tensor(y).view(-1,1)
        origin_len = torch.tensor(len(scores))
 
       
        # if self.encode_type == 'onehot':
        #     x = onehot_encode(seq)

        # elif self.encode_type == 'onehot_meiler':
        #     x = onehot_meiler_encode(seq)

        # elif self.encode_type == 'protbert':
        #     ## encode one by one, but this one take so much time, 
        #     ## use this, if we finetune model. else just pre-embed them
        #     # x = prepare_protbert(seq)

        #     x = self.encoded_seqs[idx]
        

        return {
            'seq': seq,
            'target_reg': y,
            'target_bin': (y>0).int(),
            'orig_len': origin_len
        }

        # scores = torch.tensor(row['scores'], dtype=torch.float)
        # # return original seq and do encoding in train
        # sample = {
        #     'seq': row['sequence'],
        #     'target_reg': scores,
        #     'target_bin': (scores>0).int(),
        #     'orig_len': len(scores)
        # }

        # return sample





# class Dataset:
#     def __init__(self, csv_file, max_seq_len=1024, encode_type='onehot'):
#         self.data = pd.read_csv(csv_file)
#         self.data['scores'] = self.data['scores'].apply(ast.literal_eval)

#         def count_pos_neg_values(lst):
#             count_pos = sum(1 for x in lst if x > 0)
#             count_neg = sum(1 for x in lst if x <= 0)
#             return count_pos, count_neg

#         # # Apply the function to create new columns
#         # self.data[['count_positive', 'count_negative']] = self.data['scores'].apply(count_pos_neg_values).apply(pd.Series)
#         # self.data['len'] = self.data['scores'].apply(lambda x: len(x))

#         # self.data['neg_to_pos_ratio'] = self.data['count_negative'] / self.data['count_positive']

#         self.max_seq_len = max_seq_len
#         self.encode_type= encode_type

#         # Pre-encode sequences if using ProtBERT
#         if self.encode_type == 'protbert':
#             print("downlaod ProtBert tokenizer and model:")
#             self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False,resume_download=None)
#             self.model = BertModel.from_pretrained("Rostlab/prot_bert",resume_download=None)
#             self.pre_encode_sequences()

    
#     def pre_encode_sequences(self):
#         # Ensure the tokenizer and model are only downloaded once
#         # tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
#         # model = BertModel.from_pretrained("Rostlab/prot_bert")

#         sequences = self.data['sequence'].tolist()
#         print(sequences[:3])
#         sequences = [" ".join(seq) for seq in sequences]
#         sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]

#         print(sequences[:3])
#         encoded_inputs = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=self.max_seq_len)
        
#         with torch.no_grad():
#             outputs = self.model(**encoded_inputs)
        
#         self.encoded_seqs = outputs["last_hidden_state"]

    

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):

#         def prepare_protbert(sequence):
#             sequence = " ".join(sequence)
#             sequence = re.sub(r"[UZOB]", "X", sequence) 
#             tokenized_input = self.tokenizer(sequence, 
#                                                 max_length = self.max_seq_len,
#                                                 padding='max_length',
#                                                 truncation=True,
#                                                 return_attention_mask=True,
#                                                 return_tensors='pt', )
#             # print(tokenized_input)
#             out = self.model(**tokenized_input)
#             return out['last_hidden_state'].squeeze(0)
#             # return tokenized_input
    
#         if idx < 0 or idx >= len(self.data):
#             raise IndexError("Index out of range")
        
#         row = self.data.iloc[idx]
#         seq = row['sequence']
#         scores = row['scores']

#         y  = scores[:self.max_seq_len] + [0] * (self.max_seq_len - len(scores))
#         y = torch.tensor(y).view(-1,1)
#         origin_len = torch.tensor(len(scores))
 
       
#         if self.encode_type == 'onehot':
#             x = onehot_encode(seq)

#         elif self.encode_type == 'onehot_meiler':
#             x = onehot_meiler_encode(seq)

#         elif self.encode_type == 'protbert':
#             ## encode one by one, but this one take so much time, 
#             ## use this, if we finetune model. else just pre-embed them
#             # x = prepare_protbert(seq)

#             x = self.encoded_seqs[idx]
        

#         return {
#             'encoded_seq': x,
#             'target_reg': y,
#             'target_bin': (y>0).int(),
#             'orig_len': origin_len
#         }

#         # scores = torch.tensor(row['scores'], dtype=torch.float)
#         # # return original seq and do encoding in train
#         # sample = {
#         #     'seq': row['sequence'],
#         #     'target_reg': scores,
#         #     'target_bin': (scores>0).int(),
#         #     'orig_len': len(scores)
#         # }

#         # return sample
