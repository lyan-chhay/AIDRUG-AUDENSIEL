import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import sys
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
import ast
from tqdm import tqdm
import pandas as pd
import re
# default values using in training
NEIGHBOUR_RADIUS = 10

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
        out = df[(df["Atom_Name"].str.strip() == "CA") & (df["Chain"].isin(chain_id))].reset_index(drop=True)
        if len(out) == 0:
            raise ValueError("No matching chain in the PDB file.")
        return out


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
        df_Calpha_chain_of_interest = df[(df["Chain"].isin(chainID)) & (df["Atom_Name"]=="CA")]
    
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
    df_Calpha_chain_of_interest = get_Calpha_df(df,chainID)

    # if chainID is None:
    #     df_Calpha_chain_of_interest = get_Calpha_df(df)
    # else:
    #     df_Calpha_chain_of_interest = df[(df["Chain"]==chainID) & (df["Atom_Name"]=="CA")]

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

def process_pdb2graph(pdb_path,graph_save_path, chain=None,len_cutoff= 500, score_in_bfactor=True):
    
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

    coors = get_coors(pdb_df,chain).float()
    coors[coors != coors] = 0  # Replace NaNs with zeros
    feats = get_all_node_features(pdb_df,chain).float()
    edges = get_edge_features(pdb_df,chain).float()
    edge_index = adjacency_matrix_to_edge_index(edges.squeeze(-1))
    
    if score_in_bfactor:
        scores = get_bfactor_from_df(pdb_df,chain)
        y = torch.tensor(scores)
    else:
        y = None


    # if len(y) > len_cutoff:

    #     # print("trucate ", len(y) , " to ", len_cutoff)
    #     ## trucate graph to just a specific size (not too big that cause GPU problem such as 2000nodes)
    #     feats = feats[:len_cutoff]

    #     # 2. Filter edge indices to include only edges between the first 500 nodes
    #     mask = (edge_index[0] < len_cutoff) & (edge_index[1] < len_cutoff)
    #     edge_index = edge_index[:, mask]

    #     coors = coors[:len_cutoff]
    #     y = y[:len_cutoff]

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


def process_pdb2graph_withseq(pdb_path,graph_save_path, chain=None,len_cutoff= 500, score_in_bfactor=True):
    
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

    coors = get_coors(pdb_df,chain).float()
    coors[coors != coors] = 0  # Replace NaNs with zeros
    feats = get_all_node_features(pdb_df,chain).float()
    edges = get_edge_features(pdb_df,chain).float()
    edge_index = adjacency_matrix_to_edge_index(edges.squeeze(-1))
    
    if score_in_bfactor:
        scores = get_bfactor_from_df(pdb_df,chain)
        y = torch.tensor(scores)
    else:
        y = None
  
    seq = get_seq_from_df(pdb_df, chain)

    graph = Data(pos=coors, edge_index=edge_index, seq= seq, y=y)
    # graph = Data(x=feats, pos=coors, edge_index=edge_index, seq= seq, y=y)
    # Save the graph
    torch.save(graph, graph_save_path)

    return graph



