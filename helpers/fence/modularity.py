import torch


def get_modularity_loss(H, V, target_dims, block_size = 512):
    """
    Computes the normalized L1 distance-weighted modularity loss for the given input tensors H and V,
     considering only interactions involving the specified target dimensions.
    
    Params:
        @H: A B x N x D hidden state tensor
        @V: The D x I weight tensor which is multiplied by H (may need to be transposed)
        @target_dims: The list of dimensions (0-indexed) within H to consider for the restricted loss calculation.

    Returns:
        The computed normalized L1 modularity loss for the restricted dimensions.
    """
    
    B, N, D = H.shape  # Batch size (B), token length (N), dimension (D)
    I = V.shape[1]     # Output size (I)

    target_dims = torch.tensor(target_dims, device = H.device)

    # Initialize the numerator and denominator
    numerator = torch.zeros(B, N, device=H.device, dtype=H.dtype)
    denominator = torch.zeros(B, N, device=H.device, dtype=H.dtype)

    # Distance matrix: |i - j| (only once, not recomputed in blocks)
    indices = torch.arange(D, device=H.device)
    distance_matrix = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0)).float()

    # Compute the modularity loss in blocks to save memory
    for block_start in range(0, D, block_size):
        block_end = min(block_start + block_size, D)
        
        # Extract the block for H and V
        H_block = H[:, :, block_start:block_end]  # Shape: (B, N, block_size)
        V_block = V[block_start:block_end, :]     # Shape: (block_size, I)

        # Compute the interaction between the current block and all dimensions
        H_interaction = torch.abs(H_block.unsqueeze(3) * H.unsqueeze(2))  # Shape: (B, N, block_size, D)
        
        # Mask out self-interactions for this block
        for i in range(block_start, block_end):
            H_interaction[:, :, i - block_start, i] = 0  # Avoid self-interaction

        # Weight the interaction by the distance and the V matrix interactions
        V_interaction = torch.abs(V_block.unsqueeze(1) * V.unsqueeze(0)).sum(dim=-1)  # Shape: (block_size, D)
        weighted_interaction = H_interaction * V_interaction * distance_matrix[block_start:block_end, :]  # Shape: (B, N, block_size, D)

        # Sum the interactions over the relevant dimensions
        numerator += weighted_interaction[:, :, target_dims, :].sum(dim=(-1, -2))
        denominator += (H_interaction * V_interaction)[:, :, target_dims, :].sum(dim=(-1, -2))

    # Normalize the loss
    loss = torch.where(denominator > 0, numerator / denominator, torch.zeros_like(denominator))

    # Sum over all batches and tokens
    return loss.sum()
