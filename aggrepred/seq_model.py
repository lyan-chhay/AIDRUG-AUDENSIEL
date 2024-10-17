import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from typing import Optional


# max length of the sequence set to 1000
SEQ_MAX_LEN = 1000

# 21 amino acids + 7 meiler features
INPUT_FEATURES = 28

# kernel size as per Parapred
KERNEL_SIZE = 11

# hidden output chanel of CNN
HIDDEN_CHANNELS = 256



def generate_mask(input_tensor: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Generate a mask for masked 1D convolution based on a binary mask, including non-consecutive valid positions.

    :param input_tensor: an input tensor for convolution (batch_size x features x max_seqlen)
    :param masks: a binary mask (batch_size x max_seqlen) indicating valid positions
    :return: mask (batch_size x features x max_seqlen)
    """
    batch_size, channels, max_seqlen = input_tensor.shape

    # Expand the binary mask to match the input tensor shape (batch_size x features x max_seqlen)
    conv_mask = masks.unsqueeze(1).expand(batch_size, channels, max_seqlen)

    return conv_mask.to(device=input_tensor.device)

class LocalExtractorBlock(nn.Module):
    def __init__(self,
                 input_dim: int = SEQ_MAX_LEN,
                 output_dim: int = SEQ_MAX_LEN,
                 in_channel: int = INPUT_FEATURES,
                 out_channel: Optional[int] = None,
                 kernel_size: int = KERNEL_SIZE,
                 dilation: int = 1,
                 stride: int = 1):
        
        super().__init__()

        # Assert same shape
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim

        self.in_channels = in_channel
        self.out_channel = in_channel if out_channel is None else out_channel


        # Determine the padding required for keeping the same sequence length
        assert dilation >= 1 and stride >= 1, "Dilation and stride must be >= 1."
        self.dilation, self.stride = dilation, stride
        self.kernel_size = kernel_size

        padding = self.determine_padding(self.input_dim, self.output_dim)

        self.conv = nn.Conv1d(
            in_channel,
            out_channel,
            self.kernel_size,
            padding=padding)

        self.BN = nn.BatchNorm1d(out_channel)
        self.leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_tensor: torch.Tensor, binary_mask) -> torch.Tensor:
        """
        Forward pass of the LocalExtractorBlock

        :param input_tensor: an input tensor of (bsz x features x seqlen) or (bsz x  x seqlen)
        :param mask: a boolean tensor of (bsz x 1 x seqlen)
        :return:
        """
  
        o = self.conv(input_tensor)
        o = self.BN(o)
        o = self.leakyrelu(o)
        o = self.dropout(o)

        #mask to zero-out values beyond the sequence length
        mask = generate_mask(o, binary_mask)

        return o * mask
    
    def determine_padding(self, input_shape: int, output_shape: int) -> int:
        """
        Determine the padding required to keep the same length of the sequence before and after convolution.

        formula :  L_out = ((L_in + 2 x padding - dilation x (kernel_size - 1) - 1)/stride + 1)

        :return: An integer defining the amount of padding required to keep the "same" padding effect
        """
        padding = (((output_shape - 1) * self.stride) + 1 - input_shape + (self.dilation * (self.kernel_size - 1))) // 2

        # Ensure padding is non-negative and output shape is consistent
        assert padding >= 0, f"Padding must be non-negative but got {padding}."
        return padding
    

def generate_attn_mask(batch_size, num_heads, max_length, masks):
    """
    Generate an attention mask from a provided binary mask.

    :param batch_size: int, size of the batch.
    :param num_heads: int, number of attention heads.
    :param max_length: int, maximum sequence length.
    :param masks: a binary mask (batch_size x max_length) indicating the valid positions.
    :return: expanded mask for multi-head attention (batch_size * num_heads x max_length x max_length)
    """
    # Initialize a 3D attention mask (batch_size x max_length x max_length)
    attn_mask = torch.zeros((batch_size, max_length, max_length), dtype=torch.bool)
    

    # Populate the attention mask based on the input binary masks
    for i, mask in enumerate(masks):
        # Use the binary mask to determine valid positions
        attn_mask[i] = torch.outer(mask, mask)
        attn_mask[i].fill_diagonal_(True)
    
    # Expand the mask for multiple attention heads
    attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
    
    # Reshape to merge batch and head dimensions
    attn_mask = attn_mask.reshape(batch_size * num_heads, max_length, max_length)

    return attn_mask

class Att_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, rnn_dropout=0.2, num_heads=1):
        super(Att_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_heads = num_heads

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=rnn_dropout if num_layers > 1 else 0)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2 if bidirectional else hidden_size, num_heads=num_heads, batch_first=True)
    
    def forward(self, x, banary_mask):
        """
        Forward pass through BiLSTM with Multi-Head Attention
        """
        # Packed sequences are not necessary since we are using a mask.
        h0 = torch.randn(2 * self.num_layers if self.bidirectional else self.num_layers,
                        x.size(0), self.hidden_size).to(x.device)
        c0 = torch.randn(2 * self.num_layers if self.bidirectional else self.num_layers,
                        x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))

        # Apply MultiHeadAttention
        mask = generate_attn_mask(x.size(0),self.num_heads, x.size(1), banary_mask).to(device=output.device)
        attn_output, attn_weight = self.attention(output, output, output, attn_mask=~mask)

        # mask = generate_attn_mask(x.size(0),self.num_heads, x.size(1), lengths).to(x.device)    #(batch_size, max_length, max_length)
        # # attn_output, attn_weight = self.attention(output, output, output)
        # attn_output, attn_weight = self.attention(output, output, output, attn_mask=~mask)

        return attn_output, (hn, cn)
    
  
    
class GlobalInformationExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, rnn_dropout=0.2, num_heads=1):
        super(GlobalInformationExtractor, self).__init__()
        self.att_bilstm = Att_BiLSTM(input_size, hidden_size, num_layers, bidirectional, rnn_dropout, num_heads)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, lengths):
        output, (hn,cn) = self.att_bilstm(x, lengths)
        output = self.leakyrelu(output)
        output = self.dropout(output)
        return output, (hn, cn)
    
class Aggrepred(nn.Module):
    """

    """
    def __init__(self, config):
        """
        Initialize the Aggrepred model using a configuration dictionary.

        :param config: A dictionary containing all the parameters for the model.
        """

        super().__init__()

        self.pooling = config.get("pooling", False) 
        self.use_local = config.get("use_local", True)  # Default to False if not in config
        self.use_global = config.get("use_global", True)
        
        num_localextractor_block = config.get("num_localextractor_block", 3)
        input_dim = config.get("input_dim", 1000)
        output_dim = config.get("output_dim", 1000)
        in_channel = config.get("in_channel", 28)
        out_channel = config.get("out_channel", None)
        kernel_size = config.get("kernel_size", 23)
        dilation = config.get("dilation", 1)
        stride = config.get("stride", 1)
        
        rnn_hid_dim = config.get("rnn_hid_dim", 256)
        rnn_layers = config.get("rnn_layers", 1)
        bidirectional = config.get("bidirectional", True)
        rnn_dropout = config.get("rnn_dropout", 0.2)
        attention_heads = config.get("attention_heads", 1)

        # assert self.use_local or self.use_global, "At least one of the local or global information extractor must be used."

        out_channel = in_channel if out_channel is None else out_channel
        
        if self.use_local:
            assert num_localextractor_block > 0, "Number of local extractor blocks must be greater than 0."
            self.local_extractors = nn.ModuleList([
                LocalExtractorBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    in_channel=in_channel if i == 0 else out_channel,
                    out_channel=out_channel,
                    kernel_size=kernel_size,
                    dilation = dilation,
                    stride = stride
                ) for i in range(num_localextractor_block)
            ])
            self.residue_map = nn.Linear(in_channel,out_channel)
        
        if self.use_global:
            self.global_extractor = GlobalInformationExtractor(input_size=in_channel, hidden_size=rnn_hid_dim, num_layers=rnn_layers, bidirectional=bidirectional, rnn_dropout=rnn_dropout, num_heads=attention_heads)

        
        rnn_hid_dim = rnn_hid_dim * 2 if bidirectional else rnn_hid_dim

        if self.use_local and self.use_global:
            fc_in_dim = out_channel + rnn_hid_dim
        elif self.use_local:
            fc_in_dim = out_channel
        elif self.use_global:
            fc_in_dim = rnn_hid_dim
        else:
            fc_in_dim = in_channel
        
        self.reg_layer = nn.Sequential(
            nn.Linear(fc_in_dim, 256),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
 

    def forward(self, input_tensor: torch.Tensor, binary_mask) -> torch.Tensor:
        """
        
        """

        #### Local Extracted Information
        # residual connection following 3 layers of local extractor blocks 

        if self.use_local:
            residue = self.residue_map(input_tensor)
            o = input_tensor.permute(0, 2, 1)
            residue = residue.permute(0, 2, 1)
            for extractor in self.local_extractors:
                o = extractor(o, binary_mask)
                o = o + residue
                residue = o

            local_extracted_info = o.permute(0, 2, 1)


        #### Local Extracted Information
        if self.use_global:
            global_extracted_info, (hn,cn) = self.global_extractor(input_tensor, binary_mask)


        if self.use_local and self.use_global:
            final_info = torch.cat((local_extracted_info, global_extracted_info), dim=-1)
        elif self.use_local:
            final_info = local_extracted_info
        elif self.use_global:
            final_info = global_extracted_info
        else:
            final_info = input_tensor

        reg_output = self.reg_layer(final_info)
        
        return final_info, reg_output
  

def clean_output(output_tensor: torch.Tensor, binary_mask: torch.Tensor) -> torch.Tensor:
    """
    Clean the output tensor of probabilities to remove the predictions for padded positions using a binary mask.

    :param output_tensor: output from the Parapred model; shape: (max_length x 1)
    :param binary_mask: binary mask for the sequence; shape: (max_length, ), where True indicates valid positions.

    :return: cleaned output tensor; shape: (sum(binary_mask), )
    """
    # Use the binary mask to filter out the padded positions
    return output_tensor[binary_mask].view(-1)

def clean_output_batch(output_tensor: torch.Tensor, binary_masks: torch.Tensor) -> torch.Tensor:
    """
    Clean the output tensor of probabilities to remove the predictions for padded positions in a batch using binary masks.

    :param output_tensor: output from the Parapred model; shape: (batch_size, max_length, 1)
    :param binary_masks: binary masks for the sequences; shape: (batch_size, max_length), where True indicates valid positions.

    :return: cleaned output tensor; shape: (sum of valid positions across the batch, )
    """
    batch_size, max_length, _ = output_tensor.shape
    cleaned_outputs = []

    # Loop over each sequence in the batch
    for i in range(batch_size):
        # Use the binary mask to filter out the padded positions for each sequence
        cleaned_outputs.append(output_tensor[i][binary_masks[i]].view(-1))

    # Concatenate the cleaned outputs from all sequences
    return torch.cat(cleaned_outputs, dim=0)
