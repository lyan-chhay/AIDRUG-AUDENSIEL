import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from aggrepred.EGNN import EGNN
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm

class GATModel(torch.nn.Module):
    def __init__(self, num_feats, graph_hidden_layer_output_dims, linear_hidden_layer_output_dims, heads=4, dropout=0.2):
        """
        Initialize the model with the given parameters, including Transformer-like skip connections and layer normalization.
        
        Args:
        - num_feats (int): Number of input features (dimensionality of the input nodes).
        - graph_hidden_layer_output_dims (list of int): List of hidden dimensions for GAT layers.
        - linear_hidden_layer_output_dims (list of int): List of hidden dimensions for fully connected layers.
        - heads (int): Number of attention heads for the GAT layers. Default is 8.
        - dropout (float): Dropout rate applied between layers. Default is 0.6.
        """
        super(GATModel, self).__init__()
        
        self.gat_layers = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        # First GAT layer (input layer)
        self.gat_layers.append(GATConv(num_feats, graph_hidden_layer_output_dims[0], heads=heads))
        self.layer_norms.append(LayerNorm(graph_hidden_layer_output_dims[0] * heads))
        
        # Hidden GAT layers with Transformer-like skip connections and layer normalization
        for i in range(1, len(graph_hidden_layer_output_dims)):
            self.gat_layers.append(GATConv(graph_hidden_layer_output_dims[i-1] * heads, graph_hidden_layer_output_dims[i], heads=heads))
            self.layer_norms.append(LayerNorm(graph_hidden_layer_output_dims[i] * heads))
        
        # Linear layers for matching dimensions in residual connections
        self.match_dim1 = nn.Linear(num_feats, graph_hidden_layer_output_dims[0] * heads)
        # self.match_dim2 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)

        # Fully Connected layers
        self.fc_layers = torch.nn.ModuleList()
        self.fc_layer_norms = torch.nn.ModuleList()
        
        for i in range(len(linear_hidden_layer_output_dims)):
            if i == 0:
                self.fc_layers.append(torch.nn.Linear(graph_hidden_layer_output_dims[-1] * heads, linear_hidden_layer_output_dims[i]))
            else:
                self.fc_layers.append(torch.nn.Linear(linear_hidden_layer_output_dims[i-1], linear_hidden_layer_output_dims[i]))
            self.fc_layer_norms.append(LayerNorm(linear_hidden_layer_output_dims[i]))
        
        # Final output layer
        self.fc_layers.append(torch.nn.Linear(linear_hidden_layer_output_dims[-1], 1))  # Output is a single value (aggregation score)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x_residual = self.match_dim1(x)

        # Pass through GAT layers with Transformer-like skip connections and layer normalization
        for i, gat_layer in enumerate(self.gat_layers):
            
            x = gat_layer(x, edge_index)
            x = x_residual + x  # Residual connection
            x = self.layer_norms[i](x)  # Layer normalization
            x = F.hardtanh(x)

            x_residual = x  # Save input for skip connection
        
        # Pass through Fully Connected layers 
        for i, fc_layer in enumerate(self.fc_layers[:-1]):
            x = fc_layer(x)
            x = self.leakyrelu(x)
            x = F.dropout(x, p=0.5)
     
        # Final output layer (no activation)
        x = self.fc_layers[-1](x)
        
        return x


class EGNN_Model(nn.Module):
    '''
    Paragraph uses equivariant graph layers with skip connections
    '''
    def __init__(
        self,
        num_feats,
        edge_dim = 1,
        output_dim = 1,
        graph_hidden_layer_output_dims = None,
        linear_hidden_layer_output_dims = None,
        update_coors = False,
        dropout = 0.0,
        m_dim = 16
    ):
        super(EGNN_Model, self).__init__()

        self.input_dim = num_feats
        self.output_dim = output_dim
        current_dim = num_feats
        
        # these will store the different layers of out model
        self.graph_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        
        # model with 1 standard EGNN and single dense layer if no architecture provided
        if graph_hidden_layer_output_dims == None: graph_hidden_layer_output_dims = [num_feats]
        if linear_hidden_layer_output_dims == None: linear_hidden_layer_output_dims = []
        
        # graph layers
        for hdim in graph_hidden_layer_output_dims:
            self.graph_layers.append(EGNN(dim = current_dim,
                                          edge_dim = edge_dim,
                                          update_coors = update_coors,
                                          dropout = dropout,
                                          m_dim = m_dim))
            current_dim = hdim
            
        # dense layers
        for hdim in linear_hidden_layer_output_dims:
            self.linear_layers.append(nn.Linear(in_features = current_dim,
                                                out_features = hdim))
            current_dim = hdim
        
        # final layer to get to per-node output
        self.linear_layers.append(nn.Linear(in_features = current_dim, out_features = output_dim))
        
        
    def forward(self, feats, coors, edges, mask=None):

        # graph layers
        for layer in self.graph_layers:
            feats = F.hardtanh(layer(feats, coors, edges, mask))
            
        # dense layers
        for layer in self.linear_layers[:-1]:
            feats = F.hardtanh(layer(feats))
            
        # output (i.e. prediction)
        feats = self.linear_layers[-1](feats)
        
        return feats
    