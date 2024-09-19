


# def encode_batch(batch_of_sequences: List[str], max_length: int = None) -> (torch.Tensor, torch.Tensor):
#     """
#     Encode a batch of sequences into tensors, along with their lengths

#     :param batch_of_sequences:
#     :param max_length:
#     :return:
#     """
#     encoded_seqs = []
#     seq_lens = []

#     for seq in batch_of_sequences:
#         encoded_seqs.append(encode_seq(seq, max_length=max_length))
#         seq_lens.append(len(seq))

#     # Convert list of Tensors into a bigger tensor
#     try:
#         encoded_seqs = torch.stack(encoded_seqs)
#     except RuntimeError:
#         # The expectation is that torch.stack should allow us to create a bigger tensor
#         # if not, we should pad the sequences.
#         encoded_seqs = torch.nn.utils.rnn.pad_sequence(encoded_seqs, batch_first=True)

#     # Parapred first applies a CNN to an input tensor.
#     # CNNs in PyTorch expect a tensor T of (Bsz x n_features x seqlen)
#     # Hence the permutation
#     encoded_seqs = encoded_seqs.permute(0, 2, 1)

#     return encoded_seqs, torch.as_tensor(seq_lens)
