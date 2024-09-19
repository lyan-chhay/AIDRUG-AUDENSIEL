
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import sys
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from tqdm import tqdm

top_folder_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, top_folder_path)


from aggrepred.utils import *

# default values using in training
NEIGHBOUR_RADIUS = 10



# ----------------
# Helper functions
# ----------------

# Dictionary to convert 3-letter codes to 1-letter codes
AA_3to1 = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}

# Dictionary to convert 1-letter codes to 3-letter codes
AA_1to3 = {v: k for k, v in AA_3to1.items()}

def get_Calpha_df(df, chain_id=None):
    '''
    Filters a DataFrame containing PDB data to return only the C-alpha atoms for specified chain IDs.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing PDB data.
    chain_id (list): List of chain IDs to filter by. Default is [None], meaning take all.
    
    Returns:
    pd.DataFrame: DataFrame containing only the C-alpha atoms for all(the whole pdb) or the specified chain IDs.
    '''
    if chain_id is None:
        return df[(df["Atom_Name"].str.strip() == "CA")].reset_index(drop=True)
    else:
        return df[(df["Atom_Name"].str.strip() == "CA") & (df["Chain"].isin(chain_id))].reset_index(drop=True)
    


def get_AA_onehot_features(df,  chain_id=None):
    '''
    Encodes CDR residues types as one-hot vectors for model input
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :returns: tensor (num_CDR_residues, 20) one-hot encoding for each 20 AA types
    '''
    
    # get CDR C-alpha atoms only
    df_CDRs = get_Calpha_df(df, chain_id)
    df_Calpha = get_Calpha_df(df, chain_id)
    df_CDRs = get_Calpha_df(df, chain_id)
    
    AA_unique_names = get_ordered_AA_3_letter_codes()
    AA_name_dict = {name: idx for idx, name in enumerate(AA_unique_names)}
    
    # nice names to make rest of code more understandable
    num_rows = df_Calpha.shape[0]
    num_AA = len(AA_unique_names)
    
    # convert AA name to one-hot encoding
    AA_onehot_matrix = np.zeros((num_rows, num_AA))
    
    # we will only non-zero elements where residues actually exist
    df_Calpha_not_null = df_Calpha[~df_Calpha["AA"].isna()]
    df_Calpha_not_null_indices = df_Calpha_not_null.index.values
    
    AA_onehot_matrix[df_Calpha_not_null_indices,
                     [AA_name_dict[residue] for residue in df_Calpha_not_null["AA"]]] = 1
    
    # convert from numpy to tensor
    AA_onehot_tensor = torch.tensor(AA_onehot_matrix)
    
    return AA_onehot_tensor


def get_seq_from_df(df, chainID=None):
    '''
    Get the full ordered amino acid seq for a protein chain
    
    :param df: imgt numbered dataframe for specific pdb entry
    :param chainID: chain ID of protein in pdb file
    :return: ordered list of str of all res nums in certain chain
    '''
    if chainID is None:
        df_Calpha_chain_of_interest = get_Calpha_df(df)
    else:
        df_Calpha_chain_of_interest = df[(df["Chain"]==chainID) & (df["Atom_Name"]=="CA")]
    
    amino_acids_3letter_list = df_Calpha_chain_of_interest["AA"].values.tolist()

        # Convert the 3-letter codes to 1-letter codes
    amino_acids_1letter_list = [AA_3to1.get(aa, 'X') for aa in amino_acids_3letter_list]
    
    # Join the list into a single string
    sequence = ''.join(amino_acids_1letter_list)

    return  sequence

def get_bfactor_from_df(df, chainID=None):
    '''
    Get the full ordered amino acid seq for a protein chain
    
    :param df: imgt numbered dataframe for specific pdb entry
    :param chainID: chain ID of protein in pdb file
    :return: ordered list of str of all res nums in certain chain
    '''
    if chainID is None:
        df_Calpha_chain_of_interest = get_Calpha_df(df)
    else:
        df_Calpha_chain_of_interest = df[(df["Chain"]==chainID) & (df["Atom_Name"]=="CA")]

    return  df_Calpha_chain_of_interest["bfactor"].values.astype(float).tolist()

def get_coors(df, chain_ids=None):
    '''
    Get CDR C-alpha atom coordinates
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :returns: tensor (num_CDR_residues, 3) with x, y, z coors of each atom
    '''
    
    # get CDR C-alpha atoms only
    df_CA = get_Calpha_df(df, chain_ids)
    
    # ensure coors are numbers
    df_CA["x"] = df_CA["x"].astype(float)
    df_CA["y"] = df_CA["y"].astype(float)
    df_CA["z"] = df_CA["z"].astype(float)

    # get coors as tensor
    coors = torch.tensor(df_CA[["x", "y", "z"]].values)

    return coors

def get_edge_features(df, chain_ids=None, neighbour_radius=NEIGHBOUR_RADIUS):
    '''
    Get tensor form of adjacency matrix for all CDR C-alpha atoms
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :param neighbour_radius: max distance in Angstroms neighbours can be
    :returns: tensor (num_CDR_residues, num_CDR_residues, 1) adj matrix 
    '''
    
    xyz_arr = get_coors(df, chain_ids).numpy()
    
    # get distances
    dist_matrix = cdist(xyz_arr, xyz_arr, 'euclidean')
    dist_tensor = torch.tensor(dist_matrix)
    
    # create adjacency matrix from distance info
    adj_matrix = torch.where(dist_tensor <= neighbour_radius, 1, 0)
    
    # remove self loops - do I want to do this???  
    adj_matrix = adj_matrix.fill_diagonal_(0, wrap=False)
    
    # adjust dimensions for model input
    adj_matrix.unsqueeze_(-1)
    
    return adj_matrix


def get_all_node_features(df, chain_ids=None):
    '''
    Get tensor features embedding Amino Acid type and corresponding chain
    for each C-alpha atom in the CDR
    
    :param H_id: heavy chain ID ('None' if not available)
    :param L_id: light chain ID ('None' if not available)
    :param df: imgt numbered dataframe for specific pdb entry
    :returns: tensor (num_CDR_residues, 76||26||22) with multi-hot encoding of selection from
              AA type (20), chain H/L (2), loop L1/.../H3 (6), and imgt num (54)
    '''

    return get_AA_onehot_features(df, chain_ids)
                        
###########################################

def adjacency_matrix_to_edge_index(adj_matrix):
    """
    Convert an adjacency matrix to an edge index representation using tensor operations.

    Args:
        adj_matrix (torch.Tensor): The adjacency matrix.

    Returns:
        torch.Tensor: The edge index representation.
    """
    # Find the indices of the non-zero elements in the adjacency matrix
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).t().contiguous()
    return edge_index


def edge_index_to_adjacency_matrix(edge_index):
    """
    Convert an edge index representation to an adjacency matrix using tensor operations.

    Args:
        edge_index (torch.Tensor): The edge index representation.
        num_nodes (int): The number of nodes in the graph.

    Returns:
        torch.Tensor: The adjacency matrix.
    """
    num_nodes = torch.max(edge_index) + 1
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    adj_matrix[edge_index[0], edge_index[1]] = 1
    return adj_matrix


################################################################################


def process_pdb2graph(pdb_path,graph_save_path, score_in_bfactor=True):

    # Check if pdb_path exists
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"The PDB file path {pdb_path} does not exist.")
    
    # Check if graph_save_path exists, if not, create the directory if possible
    save_dir = os.path.dirname(graph_save_path)
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except Exception as e:
            raise OSError(f"Failed to create directory {save_dir}: {e}")

    pdb_df = format_pdb(pdb_path)

    coors = get_coors(pdb_df).float()
    coors[coors != coors] = 0  # Replace NaNs with zeros
    feats = get_all_node_features(pdb_df).float()
    edges = get_edge_features(pdb_df).float()
    edge_index = adjacency_matrix_to_edge_index(edges.squeeze(-1))
    
    if score_in_bfactor:
        scores = get_bfactor_from_df(pdb_df)
        y = torch.tensor(scores)
    else:
        y = None

    graph = Data(x=feats, pos=coors, edge_index=edge_index, y=y)
    # Save the graph
    torch.save(graph, graph_save_path)

    return graph

def get_extra_info3D(pdb_path):

    # Check if pdb_path exists
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"The PDB file path {pdb_path} does not exist.")
    
    
    pdb_df = format_pdb(pdb_path)
    # extras data that may be useful in further analysis
    df_CA = get_Calpha_df(pdb_df)

    AAs = ['' if AA is np.nan else AA for AA in df_CA["AA"].values.tolist()]
    AtomNum = ['' if num is np.nan else num for num in df_CA["Atom_Num"].values.tolist()]
    chain = df_CA["Chain"].values.tolist()
    # chain_type = ["H" if ID == H_id else "L" for ID in chain]

    # catch 'None' values and convert to string - unable to have None values in batches
    # this happens when there is only one string present
    chain = [str(chain_id) for chain_id in chain]
    IMGT = df_CA["Res_Num"].values.tolist()
    x = ['' if x is np.nan else x for x in df_CA["x"].values.tolist()]
    y = ['' if y is np.nan else y for y in df_CA["y"].values.tolist()]
    z = ['' if z is np.nan else z for z in df_CA["z"].values.tolist()]
    extras = ( AAs, AtomNum, chain, IMGT, x, y, z)


# def df2dict(df):
#     dict = {column: df[column].tolist() for column in df.columns}
#     return dict



