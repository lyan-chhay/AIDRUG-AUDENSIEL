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
from functions import *

top_folder_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, top_folder_path)

# Set the MKL_THREADING_LAYER environment variable to 'GNU'
os.environ['MKL_THREADING_LAYER'] = 'GNU'


# Streamlit app
def main():
    seed_everything(seed=42)
    st.set_page_config(page_title="Aggrepred", layout="wide")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######################
    ## load protein model, load antibody model
    ######################
    seq_weight_path = "weights/protein/seq/onehot_meiler"
    # seq_weight_path = "weights/protein/seq/esm35M"
    seq_model, seq_config = define_load_seq_model(seq_weight_path)  
    seq_model = seq_model.to(device)

    graph_weight_path = "weights/protein/graph"
    graph_model, graph_config = define_load_graph_model(graph_weight_path)
    graph_model = graph_model.to(device)
 
    seq_weight_path = "weights/antibody/onehot_meiler"
    antibody_seq_model, antibody_config = define_load_seq_model(seq_weight_path)  
    antibody_seq_model = antibody_seq_model.to(device)

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
    st.title("AggrePred: Aggregation Scores Prediction Tool for Proteins/Antibody")

    if "fasta_text" not in st.session_state:
        st.session_state.fasta_text = ""


    # Sidebar for tabs
    with st.sidebar:
        st.title("Navigation")
        selected_tab = st.radio("Go to", [ "Antibodies","Proteins"], label_visibility="collapsed")
        # selected_tab = st.radio("Go to", ["General Proteins", "Antibodies"])

    if selected_tab == "Antibodies":
        if "heavy_chain" not in st.session_state:
            st.session_state.heavy_chain = ""
        if "light_chain" not in st.session_state:
            st.session_state.light_chain = ""
  

        print("")
        heavy_chain = st.text_area("Enter Heavy Chain Sequence", value=st.session_state.heavy_chain, height=100)
        if st.button("Load Example Heavy Chain"):
            st.session_state.heavy_chain = '''QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCAR
            '''
            heavy_chain = st.session_state.heavy_chain
            st.experimental_rerun() 
        
        light_chain = st.text_area("Enter Light Chain Sequence", value=st.session_state.light_chain, height=100)
        if st.button("Load Example Light Chain"):
            st.session_state.light_chain = '''EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDL
            '''
            light_chain = st.session_state.light_chain
            st.experimental_rerun() 

        # Add the 'Auto-Mutate' checkbox
        auto_mutate = st.checkbox("Auto-Mutate",help="Automatically mutate the protein to reduce overall aggregation.")
        use_ddg = st.checkbox("Calculate Energy Change", help="This will calculate the energy change (ΔΔG) upon mutation. \nNote: It may take more time for large proteins or if there are many positive residues.")
        threshold =  st.slider("Select a value", min_value=0.0, max_value=2.0, step=0.1,help="This is the threshold you want to mutate the residue. The default value is 0.")

        if st.button("Predict"):
            if heavy_chain and light_chain:
                seed_everything(seed=42)
                torch.cuda.empty_cache()
                seq_model.eval()
                print("Model in eval mode:", seq_model.training)
            
                results_df = perform_prediction_antibody(antibody_seq_model,heavy_chain,light_chain,antibody_config,device)

            
                ##mutate the residue
                if auto_mutate:
                    mutate_result_df = perform_auto_mutate(True,results_df,antibody_seq_model,antibody_config, device,wild_fasta_path,mutate_fasta_path,var_path,temp_THPLM_dir,embed_dir,use_ddg ,threshold)
                           
                    
                st.subheader('Predicted Aggregation Score:')

                tab1, tab2 = st.tabs(["🗃 Aggregation Score Predictions", " 📈 Aggregation Score Plots"])

                with tab1:
                    st.subheader(f'Predicted Aggregation Score Table:')
                    display_aggregation_table(results_df)

                with tab2:
                    st.subheader(f'Predicted Aggregation Score Plot:')
                    for protein in results_df['Protein'].unique():
                        plot_protein_values(results_df[results_df['Protein'] == protein])
                    if auto_mutate:
                        st.subheader("Recommendation for new mutants to reduce aggregation:")
                        st.dataframe(mutate_result_df.head(10))
                
            else:
                st.warning("Please enter both the heavy chain and light chain sequences.")

    elif selected_tab == "Proteins":

        input_type = st.radio("Choose input type:", ("Sequence-based", "Structure-based"), key="protein_input_type")
        
        if input_type == "Sequence-based":
            input_method = st.radio("Choose input method:", ("Text Input", "FASTA File"), key="protein_input_method")
            
            if input_method == "Text Input":
                fasta_text = st.text_area("Enter FASTA formatted text:", value=st.session_state.fasta_text, height=100)
                if st.button("Load Example"):
                    st.session_state.fasta_text = ">AF-21861\nMDTQKDVQPPKQQPMIYICGECHTENEIKSRDPIRCRECGYRIMYKKRTKRLVVFDAR"
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
            auto_mutate = st.checkbox("Auto-Mutate",help="Automatically mutate the protein to reduce overall aggregation.")
            use_ddg = st.checkbox("Calculate Energy Change", help="This will calculate the energy change (ΔΔG) upon mutation. \nNote: It may take more time for large proteins or if there are many positive residues.")
            threshold =  st.slider("Select a value", min_value=0.0, max_value=2.0, step=0.1,help="This is the threshold you want to mutate the residue. The default value is 0.")


            if st.button("Predict"):
                if auto_mutate and len(protein_sequences)>1:
                    st.warning("Please enter only 1 sequence to use auto-mutate function.")
                else:

                    seed_everything(seed=42)
                    torch.cuda.empty_cache()
                    seq_model.eval()
                    print("Model in eval mode:", seq_model.training)

                    if protein_sequences:
                        print("Starting prediction")
                        results_df = perform_prediction(seq_model, protein_sequences, headers, seq_config, device)

       
                        ##mutate the residue
                        if auto_mutate:
                            print("Starting auto-mutate")

                            mutate_result_df = perform_auto_mutate(False,results_df,seq_model,seq_config, device,wild_fasta_path,mutate_fasta_path,var_path,temp_THPLM_dir,embed_dir,use_ddg,threshold)
                            # st.write(mutate_result_df)

                        ## display
                        st.subheader('Predicted Aggregation Score:')

                        tab1, tab2 = st.tabs(["🗃 Aggregation Score Predictions", " 📈 Aggregation Score Plots"])
                        with tab1:
                            st.subheader(f'Predicted Aggregation Score Table:')
                            display_aggregation_table(results_df)

                        with tab2:
                            st.subheader(f'Predicted Aggregation Score Plot:')
                            for protein in results_df['Protein'].unique():
                                plot_protein_values(results_df[results_df['Protein'] == protein])
                            if auto_mutate:
                                st.subheader("Recommendation for new mutants to reduce aggregation:")
                                st.dataframe(mutate_result_df.head(10))

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
                    print("Model in eval mode:", graph_model.training)
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
                            st.subheader(f'Predicted Aggregation Score Table:')
                            display_aggregation_table(results_df)

                        with tab2:
                            st.subheader(f'Predicted Aggregation Score Plot:')
                            for protein in results_df['Protein'].unique():
                                plot_protein_values(results_df[results_df['Protein'] == protein])
                        
                    else:
                        st.warning("No graph created, something wrong with pdb file.")



if __name__ == "__main__":
    main()


