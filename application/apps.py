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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as graphDataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from aggrepred.model import Aggrepred, clean_output
from aggrepred.utils import *
from aggrepred.graph_utils import *
from aggrepred.graph_model import EGNN_Model, GATModel

top_folder_path = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
sys.path.insert(0, top_folder_path)


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

    if config['model'] == 'GAT':
        model = GATModel(num_feats=config["num_feats"],
                         graph_hidden_layer_output_dims=config["graph_hidden_layer_output_dims"],
                         linear_hidden_layer_output_dims=config["linear_hidden_layer_output_dims"])
    else:
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

# Streamlit app
def main():
    seed_everything(seed=42)
    st.set_page_config(page_title="Aggrepred", layout="wide")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    ######################
    ## load protein model
    ######################


    seq_weight_path = "../aggrepred/weights/seq/(onehot_meiler)_(combinedloss)_(local_3block256dim)_(global_1layer128_4head)"
    # seq_weight_path = "../aggrepred/weights/seq/(esm35M)_(combinedloss)_(local_3block256dim)_(global_1layer128_4head)"
    seq_model, seq_config = define_load_seq_model(seq_weight_path)  
    seq_model = seq_model.to(device)

    graph_weight_path = "../aggrepred/weights/graph/(onehot)_(3EGNN)"
    graph_model, graph_config = define_load_graph_model(graph_weight_path)
    graph_model = graph_model.to(device)

    ######################
    ## load antibody model
    #####################
    seq_weight_path = "../aggrepred/weights_antibody/seq/(onehot_meiler)_(combinedloss)_(local_3block256dim)_(global_1layer128_4head)"
   
    antibody_seq_model, antibody_config = define_load_seq_model(seq_weight_path)  
    antibody_seq_model = antibody_seq_model.to(device)

    #####################

    st.title("Aggrepred: Predict Aggregation Scores for Proteins")

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
            st.session_state.heavy_chain = '''QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSSASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTKTYTCNVDHKPSNTKVDKRVESKYGPPCPPCPAPEFLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSQEDPEVQFNWYVDGVEVHNAKTKPREEQFNSTYRVVSVLTVLHQDWLNGKEYKCKVSNKGLPSSIEKTISKAKGQPREPQVYTLPPSQEEMTKNQVSLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSRLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLS
            '''
            heavy_chain = st.session_state.heavy_chain
            st.experimental_rerun() 
        
        light_chain = st.text_area("Enter Light Chain Sequence", value=st.session_state.light_chain, height=100)
        if st.button("Load Example Light Chain"):
            st.session_state.light_chain = '''EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC
            '''
            light_chain = st.session_state.light_chain
            st.experimental_rerun() 

        if st.button("Predict Aggregation Score"):
            if heavy_chain and light_chain:
                seed_everything(seed=42)
                torch.cuda.empty_cache()
                seq_model.eval()
                print("Model in eval mode:", seq_model.training)
                # dataloader = DataLoader(protein_sequences, batch_size=32, shuffle=False, collate_fn=lambda x: x)

            
                results_df = pd.DataFrame(columns=['Protein', 'Position', 'Amino Acid', 'Predicted Value'])
        
                with torch.no_grad():
                    print(len(heavy_chain))
                    print(len(light_chain))

                    H_mask = torch.zeros(450, dtype=torch.bool)
                    H_mask[:len(heavy_chain)] = True  # Set the first 'len(Hchain_scores)' to 1

                    L_mask = torch.zeros(250, dtype=torch.bool)
                    L_mask[:len(light_chain)] = True  # Set the first 'len(Lchain_scores)' to 1

                    masks = torch.cat((H_mask, L_mask ), dim=0).unsqueeze(0)
                    

                    # Hchain_x = onehot_meiler_encode(heavy_chain, 450).to(device)
                    # Lchain_x = onehot_meiler_encode(light_chain, 250).to(device)

                    if antibody_config['encode_mode'] == 'esm':
                        esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                        # esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
                        # esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                        esm_model = esm_model.eval()  # Set the model to evaluation mode

                        
                        Hchain_x = embed_esm_batch([heavy_chain],  esm_model, alphabet).to(device)
                        Lchain_x = embed_esm_batch([light_chain],  esm_model, alphabet).to(device)
                        Hchain_x   = F.pad(Hchain_x, (0, 0, 0, max(450 - Hchain_x.size(1), 0)))
                        Lchain_x   = F.pad(Lchain_x, (0, 0, 0, max(250 - Lchain_x.size(1), 0)))
                        
                    elif antibody_config['encode_mode'] == 'protbert':
                        protbert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
                        protbert_model = BertModel.from_pretrained("Rostlab/prot_bert").to('cuda')

                        Hchain_x = embed_protbert_batch([heavy_chain],  protbert_model, protbert_tokenizer).to(device)
                        Lchain_x = embed_protbert_batch([light_chain],  protbert_model, protbert_tokenizer).to(device)
                        Hchain_x   = F.pad(Hchain_x, (0, 0, 0, max(450 - Hchain_x.size(1), 0)))
                        Lchain_x   = F.pad(Lchain_x, (0, 0, 0, max(250 - Lchain_x.size(1), 0)))
                    
                    elif antibody_config['encode_mode'] == 'onehot':
                        Hchain_x = onehot_encode_batch([heavy_chain], 450).to(device)
                        Lchain_x = onehot_encode_batch([light_chain], 250).to(device)
                    else:
                        Hchain_x = onehot_meiler_encode_batch([heavy_chain], 450).to(device)
                        Lchain_x = onehot_meiler_encode_batch([light_chain], 250).to(device)
                    
        
                    x = torch.cat((Hchain_x, Lchain_x), dim=1)
                    
                    print(masks.size())
                    print(x.size())
                        
                    _, out = antibody_seq_model(x, masks)
                    
                    print(out.size())

          
                    new_rows = []

                    # Clean the output for both chains
                    cleaned_output_heavy = clean_output(out[0][:450], H_mask)  # Heavy chain
                    cleaned_output_light = clean_output(out[0][450:], L_mask)  # Light chain

                    # Populate the results for both chains
                    for chain, cleaned_output, seq, label in [(heavy_chain, cleaned_output_heavy, heavy_chain, 'Heavy Chain'),
                                                            (light_chain, cleaned_output_light, light_chain, 'Light Chain')]:
                        for pos, (aa, value) in enumerate(zip(seq, cleaned_output)):
                            new_rows.append({
                                'Protein': label,
                                'Position': pos + 1,
                                'Amino Acid': aa,
                                'Predicted Value': value.item()
                            })

                    results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)
                    
                    st.subheader('Predicted Aggregation Score:')

                    tab1, tab2 = st.tabs(["🗃 Aggregation Score Predictions", " 📈 Aggregation Score Plots"])
                    
                    with tab1:
                        st.subheader(f'Predicted Aggregation Score Table:')
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download the table (CSV)",
                            data=csv,
                            file_name='prediction_results.csv',
                            mime='text/csv',
                        )
                    
                        css = """
                        <style>
                            .highlight {
                                background-color: #ffffcc;
                            }
                        </style>
                        """
                        st.markdown(css, unsafe_allow_html=True)
                
                        def highlight_positive(s):
                            if s['Predicted Value'] > 0:
                                return ['background-color: #ffff99'] * len(s)
                            else:
                                return [''] * len(s)
                        
                        for protein in results_df['Protein'].unique():
                            st.subheader(f"{protein}:")
                            to_print_df = results_df[results_df['Protein'] == protein][['Position', 'Amino Acid', 'Predicted Value']]
                            styled_df = to_print_df.set_index(to_print_df.columns[0]).style.apply(highlight_positive, axis=1)
                            st.dataframe(styled_df, use_container_width=True)
                            
                    with tab2:
                        st.subheader(f'Predicted Aggregation Score Plot:')
                        for protein in results_df['Protein'].unique():
                            plot_protein_values(results_df[results_df['Protein'] == protein])


                ####################################
                ####################################
                
                ####################################
                ####################################
            else:
                st.warning("Please enter both the heavy chain and light chain sequences.")

    elif selected_tab == "Proteins":
        # with st.expander("General Proteins", expanded=True):
        #     st.header("General Proteins")


        input_type = st.radio("Choose input type:", ("Sequence-based", "Structure-based"), key="protein_input_type")
        
        if input_type == "Sequence-based":
            input_method = st.radio("Choose input method:", ("Text Input", "FASTA File"), key="protein_input_method")
            
            if input_method == "Text Input":
                fasta_text = st.text_area("Enter FASTA formatted text:", value=st.session_state.fasta_text, height=100)
                if st.button("Load Example"):
                    st.session_state.fasta_text = ">DNA-directed RNA polymerases I II and III subunit RPABC4\nMDTQKDVQPPKQQPMIYICGECHTENEIKSRDPIRCRECGYRIMYKKRTKRLVVFDAR"
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

            if st.button("Predict"):
                seed_everything(seed=42)
                torch.cuda.empty_cache()
                seq_model.eval()
                print("Model in eval mode:", seq_model.training)
                dataloader = DataLoader(protein_sequences, batch_size=32, shuffle=False, collate_fn=lambda x: x)
                
                esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                # esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
                # esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                esm_model = esm_model.eval()  # Set the model to evaluation mode

                # protbert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
                # protbert_model = BertModel.from_pretrained("Rostlab/prot_bert").to('cuda')


                if protein_sequences:
                    results_df = pd.DataFrame(columns=['Protein', 'Position', 'Amino Acid', 'Predicted Value'])
            
                    with torch.no_grad():
                        for i, batch in enumerate(dataloader):

                            lengths = [len(seq) for seq in batch]
                            lengths = torch.tensor(lengths).to(device)
                                
                            masks = create_mask(lengths)

                            # print(lengths )

                            # print(masks)

                            if seq_config["encode_mode"] == 'esm':
                                x = embed_esm_batch(batch, esm_model, alphabet).to(device)
                            elif seq_config["encode_mode"] == 'protbert':
                                x = embed_protbert_batch(batch, protbert_model, protbert_tokenizer).to(device)
                            elif seq_config["encode_mode"] == 'onehot':
                                x = onehot_encode_batch(batch).to(device)
                            else:
                                x = onehot_meiler_encode_batch(batch, max(lengths)).to(device)

                            # x = onehot_meiler_encode_batch(batch,1000).to(device)
                            
                            _, out = seq_model(x, masks)

                            # print(out)
                            
                            new_rows = []
                            for j, sequence in enumerate(batch):
                                cleaned_output = clean_output(out[j], masks[j])
                                print(clean_output)
                                for k, value in enumerate(cleaned_output):
                                    new_rows.append({
                                        'Protein': headers[j],
                                        'Position': k+1,
                                        'Amino Acid': sequence[k],
                                        'Predicted Value': value.item()
                                    })
                            
                            new_df = pd.DataFrame(new_rows)
                            results_df = pd.concat([results_df, new_df], ignore_index=True)
                        
                        st.subheader('Predicted Aggregation Score:')

                        tab1, tab2 = st.tabs(["🗃 Aggregation Score Predictions", " 📈 Aggregation Score Plots"])
                        
                        with tab1:
                            st.subheader(f'Predicted Aggregation Score Table:')
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download the table (CSV)",
                                data=csv,
                                file_name='prediction_results.csv',
                                mime='text/csv',
                            )
                        
                            css = """
                            <style>
                                .highlight {
                                    background-color: #ffffcc;
                                }
                            </style>
                            """
                            st.markdown(css, unsafe_allow_html=True)
                    
                            def highlight_positive(s):
                                if s['Predicted Value'] > 0:
                                    return ['background-color: #ffff99'] * len(s)
                                else:
                                    return [''] * len(s)
                            
                            for protein in results_df['Protein'].unique():
                                st.subheader(f"{protein}:")
                                to_print_df = results_df[results_df['Protein'] == protein][['Position', 'Amino Acid', 'Predicted Value']]
                                styled_df = to_print_df.set_index(to_print_df.columns[0]).style.apply(highlight_positive, axis=1)
                                st.dataframe(styled_df, use_container_width=True)
                                
                        with tab2:
                            st.subheader(f'Predicted Aggregation Score Plot:')
                            for protein in results_df['Protein'].unique():
                                plot_protein_values(results_df[results_df['Protein'] == protein])

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
                                if config['model'] == 'EGNN':
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

                        st.subheader('Predicted Aggregation Score:')

                        tab1, tab2 = st.tabs(["Aggregation Score Predictions", "Aggregation Score Plots"])
                        
                        with tab1:
                            st.subheader(f'Predicted Aggregation Score Table')
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download the table (CSV)",
                                data=csv,
                                file_name='prediction_results.csv',
                                mime='text/csv',
                            )
                        
                            css = """
                            <style>
                                .highlight {
                                    background-color: #ffffcc;
                                }
                            </style>
                            """
                            st.markdown(css, unsafe_allow_html=True)
                            
                            def highlight_positive(s):
                                if s['Predicted Value'] > 0:
                                    return ['background-color: #ffff99'] * len(s)
                                else:
                                    return [''] * len(s)
                            
                            st.dataframe(results_df[['Position', 'Amino Acid', 'Predicted Value']].style.apply(highlight_positive, axis=1), use_container_width=True)

                        with tab2:
                            for protein in results_df['Protein'].unique():
                                st.subheader("Predicted Aggregation Score Plot")
                                to_print_df = results_df[results_df['Protein'] == protein][['Position', 'Amino Acid', 'Predicted Value']]
                                plot_protein_values(results_df[results_df['Protein'] == protein])
                    
                    else:
                        st.warning("No graph created, something wrong with pdb file.")




if __name__ == "__main__":
    main()