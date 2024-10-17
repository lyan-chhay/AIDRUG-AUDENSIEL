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

def embed_antiberty_batch(sequences, model):
    
    embeddings = model.embed(sequences)
    embeddings = [t[1:-1, :] for t in embeddings]  # Removes the first and last rows

    # # Pad the trimmed tensors and stack them
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings , batch_first=True)

    return embeddings
###############################################

