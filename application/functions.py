
import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import torch
import sys 
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
from Bio import SeqIO
import json
import plotly.express as px
import esm
import torch.nn.functional as F
import subprocess
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as graphDataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from aggrepred.model import Aggrepred, clean_output
from aggrepred.utils import *
from aggrepred.graph_utils import *
from aggrepred.graph_model import EGNN_Model, GATModel

top_folder_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, top_folder_path)

# Set the MKL_THREADING_LAYER environment variable to 'GNU'
os.environ['MKL_THREADING_LAYER'] = 'GNU'


## function to prediction the aggregation
def perform_prediction(seq_model, protein_sequences, headers, seq_config, device):
    results_df = pd.DataFrame(columns=['Protein', 'Position', 'Amino Acid', 'Predicted Value'])
    
    dataloader = DataLoader(protein_sequences, batch_size=32, shuffle=False, collate_fn=lambda x: x)
    
    esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    esm_model = esm_model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            lengths = [len(seq) for seq in batch]
            lengths = torch.tensor(lengths).to(device)
            masks = create_mask(lengths)

            if seq_config["encode_mode"] == 'esm':
                x = embed_esm_batch(batch, esm_model, alphabet).to(device)
            else:
                x = onehot_meiler_encode_batch(batch, max(lengths)).to(device)

            _, out = seq_model(x, masks)

            new_rows = []
            for j, sequence in enumerate(batch):
                cleaned_output = clean_output(out[j], masks[j])
                for k, value in enumerate(cleaned_output):
                    new_rows.append({
                        'Protein': headers[j],
                        'Position': k + 1,
                        'Amino Acid': sequence[k],
                        'Predicted Value': value.item()
                    })

            new_df = pd.DataFrame(new_rows)
            results_df = pd.concat([results_df, new_df], ignore_index=True)

    return results_df

def perform_prediction_antibody(antibody_seq_model,heavy_chain,light_chain,antibody_config,device):
    results_df = pd.DataFrame(columns=['Protein', 'Position', 'Amino Acid', 'Predicted Value'])
        
    with torch.no_grad():
        print(len(heavy_chain))
        print(len(light_chain))

        H_mask = torch.zeros(450, dtype=torch.bool)
        H_mask[:len(heavy_chain)] = True  # Set the first 'len(Hchain_scores)' to 1

        L_mask = torch.zeros(250, dtype=torch.bool)
        L_mask[:len(light_chain)] = True  # Set the first 'len(Lchain_scores)' to 1

        masks = torch.cat((H_mask, L_mask ), dim=0).unsqueeze(0)
    

        if antibody_config['encode_mode'] == 'esm':
            esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            esm_model = esm_model.eval()  # Set the model to evaluation mode

            Hchain_x = embed_esm_batch([heavy_chain],  esm_model, alphabet).to(device)
            Lchain_x = embed_esm_batch([light_chain],  esm_model, alphabet).to(device)
            Hchain_x   = F.pad(Hchain_x, (0, 0, 0, max(450 - Hchain_x.size(1), 0)))
            Lchain_x   = F.pad(Lchain_x, (0, 0, 0, max(250 - Lchain_x.size(1), 0)))
        
        elif antibody_config['encode_mode'] == 'onehot':
            Hchain_x = onehot_encode_batch([heavy_chain], 450).to(device)
            Lchain_x = onehot_encode_batch([light_chain], 250).to(device)
        else:
            Hchain_x = onehot_meiler_encode_batch([heavy_chain], 450).to(device)
            Lchain_x = onehot_meiler_encode_batch([light_chain], 250).to(device)
        

        x = torch.cat((Hchain_x, Lchain_x), dim=1)

        _, out = antibody_seq_model(x, masks)
        
        new_rows = []

        # Clean the output for both chains
        cleaned_output_heavy = clean_output(out[0][:450], H_mask)  # Heavy chain
        cleaned_output_light = clean_output(out[0][450:], L_mask)  # Light chain

        # Populate the results for both chains
        for chain, cleaned_output, seq, label in [(heavy_chain, cleaned_output_heavy, heavy_chain, 'Heavy'),
                                                (light_chain, cleaned_output_light, light_chain, 'Light')]:
            for pos, (aa, value) in enumerate(zip(seq, cleaned_output)):
                new_rows.append({
                    'Protein': label,
                    'Position': pos + 1,
                    'Amino Acid': aa,
                    'Predicted Value': value.item()
                })

        results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)

    return results_df


def perform_prediction_antibody_multiple(antibody_seq_model,heavy_chains,light_chains,antibody_config,device):
    results_df = pd.DataFrame(columns=['Protein', 'Chain','Position', 'Amino Acid', 'Predicted Value'])
        
    with torch.no_grad():

        heavy_lens = [len(chain) for chain in heavy_chains]
        light_lens = [len(chain) for chain in light_chains]

        H_masks = create_mask(heavy_lens ,450)
        L_masks = create_mask(light_lens,250)

        print(H_masks.size())
        print(L_masks.size())

        masks = torch.cat((H_masks, L_masks ), dim=1)
        print(masks.size())
    
        if antibody_config['encode_mode'] == 'esm':
            esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            esm_model = esm_model.eval()  # Set the model to evaluation mode

            Hchain_x = embed_esm_batch([heavy_chains],  esm_model, alphabet).to(device)
            Lchain_x = embed_esm_batch([light_chains],  esm_model, alphabet).to(device)
            Hchain_x   = F.pad(Hchain_x, (0, 0, 0, max(450 - Hchain_x.size(1), 0)))
            Lchain_x   = F.pad(Lchain_x, (0, 0, 0, max(250 - Lchain_x.size(1), 0)))
        
        elif antibody_config['encode_mode'] == 'onehot':
            Hchain_x = onehot_encode_batch([heavy_chains], 450).to(device)
            Lchain_x = onehot_encode_batch([light_chains], 250).to(device)
        else:
            Hchain_x = onehot_meiler_encode_batch([heavy_chains], 450).to(device)
            Lchain_x = onehot_meiler_encode_batch([light_chains], 250).to(device)
        

        x = torch.cat((Hchain_x, Lchain_x), dim=1)

        _, out = antibody_seq_model(x, masks)
        
        new_rows = []

        # Process the predictions for each pair of heavy and light chains in the batch
        for i, (heavy_chain, light_chain) in enumerate(zip(heavy_chains, light_chains)):
            cleaned_output_heavy = clean_output(out[i][:450], H_masks[i])  # Heavy chain for sample i
            cleaned_output_light = clean_output(out[i][450:], L_masks[i])  # Light chain for sample i

            # Populate the results for heavy and light chains of each sample
            for chain, cleaned_output, seq, label in [(heavy_chain, cleaned_output_heavy, heavy_chain, 'Heavy Chain'),
                                                      (light_chain, cleaned_output_light, light_chain, 'Light Chain')]:
                chain_type = 'Heavy' if label == 'Heavy Chain' else 'Light'
                for pos, (aa, value) in enumerate(zip(seq, cleaned_output)):
                    new_rows.append({
                        'Protein': f'Antibody_{i + 1}',  
                        'Chain': chain_type,  
                        'Position': pos + 1,
                        'Amino Acid': aa,
                        'Predicted Value': value.item()
                    })

        # Append the rows to the final DataFrame
        results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)

    return results_df


def select_mutant_residue(result_df, alpha=0.5):
    # Initialize lists to store results
    wild_sequence = None
    mutated_sequences = []
    mutation_variants = []
    mutated_headers = []

    # Group by protein (header)
    for header, group in result_df.groupby('Protein'):
        wild_sequence = group['Amino Acid'].str.cat()
        mutated_sequence = list(wild_sequence)
        variants = []

        # Iterate over each row in the group
        for _, row in group.iterrows():
            if row['Predicted Value'] > alpha:
                position = row['Position'] - 1  # Convert 1-based position to 0-based index
                original_aa = row['Amino Acid']
                mutations = ['E', 'K', 'D', 'R']
                mutations = [m for m in mutations if m != original_aa]  # Exclude the original amino acid

                for m in mutations:
                    mutated_sequence[position] = m
                    mutated_seq_str = ''.join(mutated_sequence)
                    mutated_sequences.append(mutated_seq_str)
                    variant = f"{original_aa}{row['Position']}{m}"
                    mutation_variants.append(variant)
                    mutated_headers.append(f"{header}_{variant}")
                    mutated_sequence[position] = original_aa  # Revert to original for next mutation

    return wild_sequence, mutated_sequences, mutation_variants,header, mutated_headers

def get_ddg(header_wild,wild_sequence,headers_mutate, mutated_sequences, mutation_variants, wild_fasta_path,mutate_fasta_path,var_path,temp_THPLM_dir,embed_dir):
    # Write wild sequence to wild_fasta_path
    with open(wild_fasta_path, 'w') as f:
        f.write(f">{header_wild}\n{wild_sequence}\n")

    # Write mutated sequences to mutate_fasta_path
    with open(mutate_fasta_path, 'w') as f:
        f.write(f">{header_wild,}\n{wild_sequence}\n")
        for header, sequence in zip(headers_mutate, mutated_sequences):
            f.write(f">{header}\n{sequence}\n")

    # Write mutation variants to var_path
    with open(var_path, 'w') as f:
        for variant in mutation_variants:
            # print(variant)
            f.write(f"{variant}\n")

    ##run the energy change with THPLM
    output_file = os.path.join(temp_THPLM_dir, "output.txt")
    command = f"python THPLM/esmscripts/extract.py esm2_t36_3B_UR50D {mutate_fasta_path}  {embed_dir} --repr_layers 36 --include mean --toks_per_batch 4096"
    exit_status = os.system(command)
    command = f"python THPLM/esmscripts/extract.py esm2_t36_3B_UR50D {wild_fasta_path}  {embed_dir} --repr_layers 36 --include mean --toks_per_batch 4096"
    exit_status = os.system(command)
    
    command = f"CUDA_VISIBLE_DEVICES=0 python THPLM/THPLM_predict.py {var_path} {wild_fasta_path} {embed_dir} {mutate_fasta_path} --gpunumber 0 --extractfile THPLM/esmscripts/extract.py > {output_file}"
    # print(command)
    exit_status = os.system(command)

    if exit_status == 0:
        # Read the output file to extract the DDG values
        with open(output_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]

            print(last_line)

        try:
            # Parse the last line as a Python dictionary # Convert the dictionary to a DataFrame
            output_dict = eval(last_line)  # Use eval since it's not strict JSON (single quotes)
            ddg_df = pd.DataFrame(list(output_dict.items()), columns=['Protein', 'DDG'])
 
        except json.JSONDecodeError:
            print("Failed to parse the output as JSON.")
    else:
        print("Command execution failed.")
    return ddg_df 


def perform_auto_mutate(antibody,results_df,seq_model,seq_config, device,wild_fasta_path,mutate_fasta_path,var_path,temp_THPLM_dir,embed_dir, use_ddg ='False' ,threshold=0):
    if antibody: 
        out_df = pd.DataFrame(columns=['Protein', 'Predicted Value'])
        
        heavy_wild_sequence, heavy_mutated_sequences, heavy_mutation_variants, heavy_header_wild, heavy_headers_mutate = select_mutant_residue(results_df[results_df['Protein']=='Heavy'], threshold)
        light_wild_sequence, light_mutated_sequences, light_mutation_variants, light_header_wild, light_headers_mutate = select_mutant_residue(results_df[results_df['Protein']=='Light'], threshold)
        
        new_rows =[]
        for i, (heavy_header_mutate, heavy_mutated_sequence, light_wild_sequence) in enumerate(zip(heavy_headers_mutate,heavy_mutated_sequences ,[light_wild_sequence]*len(heavy_mutated_sequences))):
            heavy_mutate_result_df = perform_prediction_antibody(seq_model,heavy_mutated_sequence, light_wild_sequence,seq_config, device)
            score_average = heavy_mutate_result_df['Predicted Value'].mean()
            new_rows.append({
                        'Protein': heavy_header_mutate,  
                        'Predicted Value': score_average
                    })
        for i, (light_header_mutate, heavy_wild_sequence, light_mutated_sequence) in enumerate(zip(light_headers_mutate,[heavy_wild_sequence]*len(light_mutated_sequences) ,light_mutated_sequences)):
            light_mutate_result_df = perform_prediction_antibody(seq_model,heavy_wild_sequence, light_mutated_sequence,seq_config, device)
            score_average = light_mutate_result_df['Predicted Value'].mean()
            new_rows.append({
                        'Protein': light_header_mutate,  
                        'Predicted Value': score_average
                    })
            
        out_df = pd.concat([out_df, pd.DataFrame(new_rows)], ignore_index=True)
        
        wild_avg_value = results_df['Predicted Value'].mean().mean()
        out_df['Score Difference'] = out_df['Predicted Value'].apply(lambda val: val - wild_avg_value)

        if use_ddg:
            ddg_df1 = get_ddg(heavy_header_wild,heavy_wild_sequence,heavy_headers_mutate,heavy_mutated_sequences, heavy_mutation_variants, wild_fasta_path,mutate_fasta_path,var_path,temp_THPLM_dir,embed_dir)
            ddg_df2 = get_ddg(light_header_wild,light_wild_sequence,light_headers_mutate,light_mutated_sequences, light_mutation_variants, wild_fasta_path,mutate_fasta_path,var_path,temp_THPLM_dir,embed_dir)
            ddg_df = pd.concat([ddg_df1, ddg_df2], axis=0)

    else:
        
        wild_sequence, mutated_sequences, mutation_variants, header_wild, headers_mutate = select_mutant_residue(results_df, 0.5)
        mutate_result_df = perform_prediction(seq_model, mutated_sequences, headers_mutate, seq_config, device)

        # Calculate average predicted value for each protein
        wild_avg_value = results_df.groupby('Protein')['Predicted Value'].mean().mean()
        out_df = mutate_result_df.groupby('Protein')['Predicted Value'].mean()
        out_df = out_df.reset_index()

        out_df['Score Difference'] = out_df['Predicted Value'] - wild_avg_value

        if use_ddg:
            ddg_df = get_ddg(header_wild,wild_sequence,headers_mutate, mutated_sequences, mutation_variants, wild_fasta_path,mutate_fasta_path,var_path,temp_THPLM_dir,embed_dir)
        
    if use_ddg:
        out_df = pd.merge(out_df, ddg_df, on='Protein')[['Protein', 'DDG', 'Score Difference']]
        final_df = out_df.sort_values(by=['DDG','Score Difference'], ascending=True)
    else:
        final_df = out_df[['Protein', 'Score Difference']].sort_values(by='Score Difference', ascending=True)

    final_df = final_df.rename(columns={'Protein': 'Mutant'})  

    return final_df

#######################
## other functions
#######################
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def create_mask(lengths, max_length=1000):
    max_length = min(max(lengths),1000)
    batch_size = len(lengths)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask

def read_fasta(fasta_text):
    headers = []
    sequences = []
    for record in SeqIO.parse(fasta_text, "fasta"):
        headers.append(record.id)
        sequences.append(str(record.seq))
    return headers, sequences



##############################################
## display table of the score
##############################################

def display_aggregation_table(results_df):
    """
    Display the predicted aggregation scores in a Streamlit app.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing the predicted aggregation scores.
    """
    
    # Allow users to download the table as a CSV file
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download the table (CSV)",
        data=csv,
        file_name='prediction_results.csv',
        mime='text/csv',
    )
    
    # CSS for highlighting positive values
    css = """
    <style>
        .highlight {
            background-color: #ffffcc;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    # Function to highlight rows with positive predicted values
    def highlight_positive(s):
        if s['Predicted Value'] > 0:
            return ['background-color: #ffff99'] * len(s)
        else:
            return [''] * len(s)
    
    # Display the table for each protein
    for protein in results_df['Protein'].unique():
        st.subheader(f"{protein}:")
        to_print_df = results_df[results_df['Protein'] == protein][['Position', 'Amino Acid', 'Predicted Value']]
        styled_df = to_print_df.set_index(to_print_df.columns[0]).style.apply(highlight_positive, axis=1)
        st.dataframe(styled_df, use_container_width=True)




def plot_protein_values(protein_df):
    fig = px.line(protein_df, x='Position', y='Predicted Value', text='Amino Acid',
                  title=f"{protein_df['Protein'].iloc[0]}")
    
    fig.update_traces(mode="lines+markers+text", textposition="bottom center",
                      hovertemplate="<br>".join([
                          "Position: %{x}",
                          "Amino Acid: %{text}",
                          "Predicted Value: %{y:.2f}"
                      ]))
    
    fig.update_yaxes(range=[-5, 5])
    
    # Add a red line at y=0
    fig.add_shape(
        type="line",
        x0=protein_df['Position'].min(), x1=protein_df['Position'].max(),
        y0=0, y1=0,
        line=dict(color="red", width=2, dash="dash")
    )

    st.plotly_chart(fig, use_container_width=True)




#######################
### load our pre-trained model
#######################

def define_load_seq_model(weight_path, device='cuda'):
    config_path = os.path.join(weight_path, "config.json")
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)

    model = Aggrepred(config)
    checkpoint_path = os.path.join(weight_path, "model_best.pt")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loading model successfully')
    else:
        print('No model found')

    return model, config

 
def define_load_graph_model(weight_path, device='cuda'):
    config_path = os.path.join(weight_path, "config.json")
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    

    model = EGNN_Model(num_feats=config["num_feats"],
                        graph_hidden_layer_output_dims=config["graph_hidden_layer_output_dims"],
                        linear_hidden_layer_output_dims=config["linear_hidden_layer_output_dims"])

    checkpoint_path = os.path.join(weight_path, "model_best.pt")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loading model successfully')
    else:
        print('No model found')

    return model, config