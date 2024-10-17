import os
import sys
import random
import json
import tempfile
import subprocess

# Data handling and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# PyTorch and related utilities
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as graphDataLoader

# BioPython for sequence handling
from Bio import SeqIO

# ESM (Evolutionary Scale Modeling)
import esm

# Streamlit for web app interface
import streamlit as st

# Custom imports from aggrepred
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from aggrepred.seq_model import Aggrepred, clean_output
from aggrepred.graph_model import EGNN_Model
from aggrepred.utils import *
from aggrepred.graph_utils import *

# Set the MKL_THREADING_LAYER environment variable to 'GNU'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Add top-level folder to system path
top_folder_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, top_folder_path)


from functions import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# Streamlit app
def main():
    seed_everything(seed=42)
    st.set_page_config(page_title="Aggrepred", layout="wide")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #####################

    # path to THPLM , in case we use 'auto-mutate'
    THPLM_dir = "THPLM/"
    temp_THPLM_dir = "THPLM/temp/"
    os.makedirs(temp_THPLM_dir , exist_ok=True)
    wild_fasta_path = os.path.join(temp_THPLM_dir , "wild.fasta")
    mutate_fasta_path = os.path.join(temp_THPLM_dir , "varlist.fasta")
    var_path = os.path.join(temp_THPLM_dir , "var.txt")
    embed_dir = os.path.join(temp_THPLM_dir , "esm3Boutput/")

    ##########################################
    ### start the application interface from here
    ##########################################
    st.title("AggrePred: Aggregation Scores Prediction Tool for Proteins and Antibodies")
    

    with st.expander("Description"):
        st.image("goal.png", caption="Prediction Overview")
        st.write("""
        ### Project Overview:
        AggrePred, aims at predicting the Aggregation Propensity of proteins and antibodies based on their sequences. The model utilizes advanced deep learning techniques to analyze sequences and provide insights into aggregation risks.
        
        ### Input Options
        You can input your protein sequence in FASTA format either by typing directly into the input field or uploading a FASTA file. 
        The file upload option allows for multiple protein predictions at once.
        
        ### Prediction
        Once the input is provided, click **Predict** to begin the analysis. Two result tabs will appear:
        
        - **AP score Plot**:  
        This tab displays a line plot showing the aggregation propensity (AP) score for each residue of the protein. 
        You can hover over the plot to view detailed information such as residue position, amino acid, and AP score.
        
        - **AP Score Table**:  
        This tab contains a table listing the AP scores. Positive AP scores (indicative of aggregation-prone residues) 
        are highlighted in yellow. There is also a download button to download the AP scores in CSV format.
        
        ### Advanced Options
        **Auto-Mutation**  
        The Auto-Mutate option automatically suggests mutations to reduce overall aggregation. 
        It replaces all residues with positive AP scores (APR) with more soluble and stable amino acids (e.g., E, D, K, R).
        
        The mutated sequence with the lowest number of APRs and average AP scores is recommended.
        
        **Free Energy Change Calculation**  
        For further analysis, you can use the **Calculate Free Energy Change** option. This feature calculates the difference in free energy 
        between the wild-type and mutated sequences. A negative value indicates that the mutated sequence is more stable than the original.
        """)
  
    if "fasta_text" not in st.session_state:
        st.session_state.fasta_text = ""


    # Sidebar for tabs
    with st.sidebar:
        st.title("Navigation")
        selected_tab = st.radio("Go to", [ "Antibodies","Proteins"], label_visibility="collapsed")
        # selected_tab = st.radio("Go to", ["General Proteins", "Antibodies"])

    if selected_tab == "Antibodies":

        ######################
        ## load protein model, load antibody model
        ######################
        print(device)
        
    
        seq_weight_path = "weights/antibody/onehot_meiler"
        antibody_seq_model, antibody_config = define_load_seq_model(seq_weight_path, device)  
        antibody_seq_model = antibody_seq_model.to(device)

        if "heavy_chain" not in st.session_state:
            st.session_state.heavy_chain = ""
        if "light_chain" not in st.session_state:
            st.session_state.light_chain = ""
  
     
        heavy_chain = st.text_area("Enter Heavy Chain Sequence", value=st.session_state.heavy_chain, height=100)
        if st.button("Load Example Heavy Chain"):
            st.session_state.heavy_chain = '''QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS'''
            heavy_chain = st.session_state.heavy_chain
            st.experimental_rerun() 
        
        light_chain = st.text_area("Enter Light Chain Sequence", value=st.session_state.light_chain, height=100)
        if st.button("Load Example Light Chain"):
            st.session_state.light_chain = '''EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEI'''
            light_chain = st.session_state.light_chain
            st.experimental_rerun() 

        st.write('Option:')
        auto_mutate = st.checkbox("Auto-Mutate",help="Automatically mutate the protein to reduce overall aggregation.")
        use_ddg = st.checkbox("Calculate Free Energy Change", help="This will calculate the energy change (ΔΔG) upon mutation. \nNote: It may take more time for large proteins or if there are many positive residues.")

        
        if st.button("Predict"):

            if heavy_chain and light_chain:
                seed_everything(seed=42)
                torch.cuda.empty_cache()
                antibody_seq_model.eval()
                
                results_df = perform_prediction_antibody(antibody_seq_model,heavy_chain,light_chain,antibody_config,device)

            
                ##mutate the residue
                if auto_mutate:
                    mutate_result_df, all_mutate_out_df = perform_auto_mutate(True,results_df,antibody_seq_model,antibody_config, device,wild_fasta_path,mutate_fasta_path,var_path,temp_THPLM_dir,embed_dir,use_ddg)

                    
                st.subheader('Predicted Aggregation Score:')

                top_k = 5

                tab1, tab2 = st.tabs(["🗃 Aggregation Score Predictions", " 📈 Aggregation Score Plots"])

                with tab1:
                    st.subheader(f'Predicted Aggregation Score Plot:')
                    
                    APR_wild_heavy_count = results_df[(results_df['Protein'] == 'Heavy') & (results_df['Predicted Value'] > 0)].shape[0]
                    APR_wild_light_count = results_df[(results_df['Protein'] == 'Light') & (results_df['Predicted Value'] > 0)].shape[0]
                    
                    for protein in results_df['Protein'].unique():
                        plot_protein_values(results_df[results_df['Protein'] == protein])

                    if auto_mutate:
                        st.subheader("Suggestions for New Mutants to Minimize Overall Aggregation:")
                        # st.dataframe(mutate_result_df.head(top_k))
                     
                        list_heavy_mutated_dfs = []
                        list_light_mutated_dfs = []

                        if use_ddg:
                            # Exclude mutants with positive DDG
                            heavy_mutants = mutate_result_df[
                                (mutate_result_df['Mutant'].str.startswith('Heavy')) & 
                                (mutate_result_df['DDG'] <= 0)
                            ].nsmallest(top_k, ['APR', 'Score Difference', 'DDG'])

                            light_mutants = mutate_result_df[
                                (mutate_result_df['Mutant'].str.startswith('Light')) & 
                                (mutate_result_df['DDG'] <= 0)
                            ].nsmallest(top_k, ['APR', 'Score Difference', 'DDG'])
                        else:
                            heavy_mutants = mutate_result_df[mutate_result_df['Mutant'].str.startswith('Heavy')].nsmallest(top_k, ['APR', 'Score Difference'])
                            light_mutants = mutate_result_df[mutate_result_df['Mutant'].str.startswith('Light')].nsmallest(top_k, ['APR', 'Score Difference'])
                        

                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Heavy Chain Point Mutation")
                            st.write("The number of APR in wild heavy chain:", APR_wild_heavy_count)
                            st.dataframe(heavy_mutants)

                        with col2:
                            st.write("Light Chain Point Mutation")
                            st.write("The number of APR in wild light chain:",APR_wild_light_count)
                            st.dataframe(light_mutants)

                        for mutant in heavy_mutants['Mutant']:
                            # Split the mutant value into protein type and variant (e.g., "heavy_V3K" -> "heavy", "V3K")
                            protein_type, variant = mutant.split('_')
                            
                            # Append DataFrame for heavy mutations
                            list_heavy_mutated_dfs.append(all_mutate_out_df[
                                (all_mutate_out_df['Protein'] == 'Heavy') & 
                                (all_mutate_out_df['variant'] == variant)
                            ])

                        for mutant in light_mutants['Mutant']:
                            # Split the mutant value into protein type and variant (e.g., "light_V4P" -> "light", "V4P")
                            protein_type, variant = mutant.split('_')
                            
                            # Append DataFrame for light mutations
                            list_light_mutated_dfs.append(all_mutate_out_df[
                                (all_mutate_out_df['Protein'] == 'Light') & 
                                (all_mutate_out_df['variant'] == variant)
                            ])                                             

                        st.subheader(f"Mutation effect comparison on light and heavy chain")
                        if list_heavy_mutated_dfs:
                            st.write('- Heavy Chain')
                            plot_protein_comparison(results_df[results_df['Protein'] == 'Heavy'], list_heavy_mutated_dfs ,'wild',mutate_result_df['Mutant'].head(top_k).tolist())
                        
                        if list_light_mutated_dfs:
                            st.write('- Light Chain')
                            plot_protein_comparison(results_df[results_df['Protein'] == 'Light'], list_light_mutated_dfs ,'wild',mutate_result_df['Mutant'].head(top_k).tolist())

                with tab2:
                    st.subheader(f'Predicted Aggregation Score Table:')
                    display_aggregation_table(results_df)

            else:
                st.warning("Please enter both the heavy chain and light chain sequences.")

    elif selected_tab == "Proteins":
        
        seq_weight_path = "weights/protein/seq/onehot_meiler"
        # seq_weight_path = "weights/protein/seq/esm35M"
        seq_model, seq_config = define_load_seq_model(seq_weight_path, device)  
        seq_model = seq_model.to(device)

        graph_weight_path = "weights/protein/graph"
        graph_model, graph_config = define_load_graph_model(graph_weight_path, device)
        graph_model = graph_model.to(device)

        input_type = st.radio("Choose input type:", ("Sequence-based", "Structure-based"), key="protein_input_type")
        
        if input_type == "Sequence-based":
            input_method = st.radio("Choose input method:", ("Text Input", "FASTA File"), key="protein_input_method")
            
            if input_method == "Text Input":
                fasta_text = st.text_area("Enter FASTA formatted text:", value=st.session_state.fasta_text, height=100)
                if st.button("Load Example"):
                    st.session_state.fasta_text = ">AF-P00441\nMATKAVCVLKGDGPVQGIINFEQKESNGPVKVWGSIKGLTEGLHGFHVHEFGDNTAGCTSAGPHFNPLSRKHGGPKDEERHVGDLGNVTADKDGVADVSIEDSVISLSGDHCIIGRTLVVHEKADDLGKGGNEESTKTGNAGSRLACGVIGIAQ"
                    fasta_text = st.session_state.fasta_text
                    st.experimental_rerun()  
                
                if fasta_text:
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as tmp_file:
                        tmp_file.write(fasta_text)
                        tmp_file_path = tmp_file.name
    

                    headers, protein_sequences = read_fasta(tmp_file_path)
                    if not protein_sequences:
                        st.warning("Please enter valid FASTA formatted text.")
                

            elif input_method == "FASTA File":
                fasta_file = st.file_uploader("Upload FASTA file", type=["fasta", "fa"])
                if fasta_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(fasta_file.read())
                        tmp_file_path = tmp_file.name
                
                    headers, protein_sequences = read_fasta(tmp_file_path)
                    os.remove(tmp_file_path)

            # Add the 'Auto-Mutate' checkbox
            st.write('Option:')
            auto_mutate = st.checkbox("Auto-Mutate",help="Automatically mutate the protein to reduce overall aggregation.")
            use_ddg = st.checkbox("Calculate Free Energy Change", help="This will calculate the energy change (ΔΔG) upon mutation. \nNote: It may take more time for large proteins or if there are many positive residues.")
            

            if st.button("Predict"):

                if any(len(seq) > 1000 for seq in protein_sequences):
                    st.warning("Protein sequences are longer than 1000 characters. Please shorten the sequences.")
                elif auto_mutate and len(protein_sequences) > 1:
                    st.warning("Please enter only 1 sequence to use auto-mutate function.")
                else:
                    seed_everything(seed=42)
                    torch.cuda.empty_cache()
                    seq_model.eval()
                    
                    if protein_sequences:
                        print("Starting prediction")
                        results_df = perform_prediction(seq_model, protein_sequences, headers, seq_config, device)

                        ##mutate the residue
                        if auto_mutate:
                            print("Starting auto-mutate")

                            mutate_result_df , all_mutate_out_df = perform_auto_mutate(False,results_df,seq_model,seq_config, device,wild_fasta_path,mutate_fasta_path,var_path,temp_THPLM_dir,embed_dir,use_ddg)
    

                        ## display
                        st.subheader('Predicted Aggregation Score:')
                        top_k =5

                        tab1, tab2 = st.tabs(["🗃 Aggregation Score Predictions", " 📈 Aggregation Score Plots"])
                        with tab1:
                            st.subheader(f'Predicted Aggregation Score Plot:')
                            for protein in results_df['Protein'].unique():
                                plot_protein_values(results_df[results_df['Protein'] == protein])

                            
                    
                            if auto_mutate:
                                st.subheader("Suggestions for New Mutants to Minimize Overall Aggregation:")

                                APR_wild_count = results_df[(results_df['Protein'] == protein) & (results_df['Predicted Value'] > 0)].shape[0]
                                st.write("The number of APR in wild sequence:",APR_wild_count)
                                st.dataframe(mutate_result_df.head(top_k),hide_index=True)
                                
                                mutated_dfs = [all_mutate_out_df[all_mutate_out_df['Protein'] == mutant] 
                                            for mutant in mutate_result_df['Mutant'].head(top_k)]
                                
                                if mutated_dfs:
                                    st.subheader(f"{protein} - Mutation Comparison")
                                    plot_protein_comparison(results_df[results_df['Protein'] == protein], mutated_dfs,'wild',mutate_result_df['Mutant'].head(top_k).tolist())

                        with tab2:
                            st.subheader(f'Predicted Aggregation Score Table:')
                            display_aggregation_table(results_df)
                            

                    else:
                        st.warning("Please enter protein sequences or upload a FASTA file.")

        else:
            pdb_file = st.file_uploader("Upload PDB file", type=["pdb"])
            if pdb_file is not None:
                pdb_file_name = pdb_file.name
                st.write(f"Uploaded file name: {pdb_file_name}")

                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(pdb_file.read())
                    tmp_file_path = tmp_file.name
                
                g = process_pdb2graph(tmp_file_path, "tmp/g.pt")
                seq = get_seq_from_df(format_pdb(tmp_file_path))

                if st.button("Predict"):
                    seed_everything(seed=42)
                    torch.cuda.empty_cache()
                    graph_model.eval()
                    # print("Model in eval mode:", graph_model.training)
                    dataloader = graphDataLoader([g], batch_size=1, shuffle=False)
                    
                    if g:
                        results_df = pd.DataFrame(columns=['Protein', 'Position', 'Amino Acid', 'Predicted Value'])

                        with torch.no_grad():
                            for idx, batch in enumerate(dataloader):
                                x, coors, edge_index, batch, y = batch.x, batch.pos, batch.edge_index, batch.batch, batch.y
                                if graph_config['model'] == 'EGNN':
                                    x = x.unsqueeze(0).to(device)
                                    edge_index = edge_index.to(device)
                                    coors = coors.unsqueeze(0).to(device)
                                    edges = edge_index_to_adjacency_matrix(edge_index).unsqueeze(2).unsqueeze(0).to(device)
                                    out = graph_model(x, coors, edges).squeeze()
                                else:
                                    x = x.to(device)
                                    edge_index = edge_index.to(device)
                                    out = graph_model(x, edge_index).squeeze()
                        
                        new_rows = []
                        for k, value in enumerate(out.tolist()):
                            new_rows.append({
                                'Protein': pdb_file_name,
                                'Position': k+1,
                                'Amino Acid': seq[k],
                                'Predicted Value': value
                            })

                        new_df = pd.DataFrame(new_rows)
                        results_df = pd.concat([results_df, new_df], ignore_index=True)

                        ## display
                        st.subheader('Predicted Aggregation Score:')

                        tab1, tab2 = st.tabs(["🗃 Aggregation Score Predictions", " 📈 Aggregation Score Plots"])
                        with tab1:
                            st.subheader(f'Predicted Aggregation Score Plot:')
                            for protein in results_df['Protein'].unique():
                                plot_protein_values(results_df[results_df['Protein'] == protein])
                            
                        with tab2:
                            st.subheader(f'Predicted Aggregation Score Table:')
                            display_aggregation_table(results_df)

                    else:
                        st.warning("No graph created, something wrong with pdb file.")



if __name__ == "__main__":
    main()


