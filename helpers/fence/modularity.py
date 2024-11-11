import torch

def get_modularity_loss_v1(H, V, target_dims):
    """
    Computes the normalized L1 distance-weighted modularity loss for the given input tensors H and V,
     considering only interactions involving the specified target dimensions.
    
    Params:
        @H: A B x N x D hidden state tensor
        @V: The D x I weight tensor which is multiplied by H (may need to be transposed)
        @target_dims: The list of dimensions (0-indexed) within H to consider for the restricted loss calculation.

    Returns:
        The computed normalized L1 modularity loss for the restricted dimensions.

    Examples:
        torch.manual_seed(0)
        H = torch.randn(10, 1024, 3072, dtype = torch.bfloat16, device = 'cuda')
        V = torch.randn(3072, 8192, dtype = torch.bfloat16, device = 'cuda')
        start = time.time()
        loss = get_modularity_loss_v1(H, V, target_dims = [3040, 3052])
        end = time.time()
        print(end - start)
        print(loss)
    """
    
    _, _, D = H.shape
    loss = torch.zeros(1, device = H.device)
    
    # Matrix multiplication to compute P = H @ V, but this is for reference, not needed directly in the loop
    # P = torch.matmul(H, V)

    # Compute the numerator of the normalized L1 loss for specified target dimensions
    for i in target_dims:
        for j in range(D):  # Interact i with all other dimensions (including non-target dimensions)
            if i != j:  # Skip self-interaction
                # Interaction term |h_i * h_j|
                interaction = torch.abs(H[:, :, i] * H[:, :, j])
                # Multiply interaction by the weight matrix V for the j-th column
                weighted_interaction = interaction * torch.abs(V[i, :] * V[j, :]).sum(dim=-1) * abs(i - j)
                # Sum over batches and tokens for the total weighted loss
                loss += weighted_interaction.sum()

    # Compute the denominator (normalization factor) which is the sum of all interaction terms
    sum_interactions = torch.zeros(1, device = H.device)
    for i in target_dims:
        for j in range(D):
            if i != j: 
                interaction = torch.abs(H[:, :, i] * H[:, :, j])
                # Sum interaction terms without distance weighting, normalized by weights in V
                unweighted_interaction = interaction * torch.abs(V[i, :] * V[j, :]).sum(dim=-1)
                sum_interactions += unweighted_interaction.sum()

    # Normalize the loss
    if sum_interactions > 0:
        loss = loss / sum_interactions

    return loss


def get_modularity_loss_v2(H, V, target_dims):
    """
    Computes the normalized L1 distance-weighted modularity loss for the given input tensors H and V,
    considering only interactions involving the specified target dimensions, while avoiding memory issues.
    
    Details:
        Optimizes `get_modularity_loss_v1` by:
        - Combining the numerator + denominator into a single loop.  
        - Calculates numerator + denominator on the fly
        - Recycles the v_interaction calculation for each pair (i, j)
        - Calculates abs value of H once
        
        Reduced benchmark time with Df = 2 from .55s (v1) -> .35s (v2)

    Params:
        @H: A B x N x D hidden state tensor
        @V: The D x I weight tensor which is multiplied by H
        @target_dims: The list of dimensions (0-indexed) within H to consider for the restricted loss calculation.

    Returns:
        The computed normalized L1 modularity loss for the restricted dimensions.

    Examples:
        torch.manual_seed(0)
        H = torch.randn(10, 1024, 3072, dtype = torch.bfloat16, device = 'cuda')
        V = torch.randn(3072, 8192, dtype = torch.bfloat16, device = 'cuda')
        start = time.time()
        loss = get_modularity_loss_v2(H, V, target_dims = [3040, 3052])
        end = time.time()
        print(end - start)
        print(loss)
    """
    
    _, _, D = H.shape
    loss = torch.zeros(1, device = H.device)
    sum_interactions = torch.zeros(1, device = H.device)
    abs_H = torch.abs(H)

    # Compute the numerator (distance-weighted) and denominator (unweighted) on the fly
    for i in target_dims:
        for j in range(D):
            if i != j: 
                # Compute interaction term |h_i * h_j * v_i * v_j| on the fly
                interaction = abs_H[:, :, i] * abs_H[:, :, j]
                v_interaction = torch.abs(V[i, :] * V[j, :]).sum(dim = -1)
                
                # Calculate distance-weighted interaction for numerator
                weighted_interaction = interaction * v_interaction * abs(i - j)

                # Sum over batches and tokens for the total weighted loss (numerator)
                loss += weighted_interaction.sum()

                # Calculate unweighted interaction for denominator
                sum_interactions += (interaction * v_interaction).sum()

    # Normalize the loss
    if sum_interactions > 0:
        loss = loss / sum_interactions

    return loss

def get_modularity_loss_v3(H, V, target_dims):
    """
    Computes the normalized L1 distance-weighted modularity loss for the given input tensors H and V,
     considering only interactions involving the specified target dimensions, while avoiding memory issues.
    
    Details:
        Optimizes `get_modularity_loss_v2` by:
        - Precalculates the V interaction matrix
        - Moves the summation over the B and N dimensions outside the loop
        
        Reduced benchmark time with Df = 2 from .55s (v1) -> .35s (v2) -> .20s (v3)

    Params:
        @H: A B x N x D hidden state tensor
        @V: The D x I weight tensor which is multiplied by H
        @target_dims: The list of dimensions (0-indexed) within H to consider for the restricted loss calculation.

    Returns:
        The computed normalized L1 modularity loss for the restricted dimensions.

    Examples:
        torch.manual_seed(0)
        H = torch.randn(10, 1024, 3072, dtype = torch.bfloat16, device = 'cuda')
        V = torch.randn(3072, 8192, dtype = torch.bfloat16, device = 'cuda')
        start = time.time()
        loss = get_modularity_loss_v3(H, V, target_dims = [3040, 3052])
        end = time.time()
        print(end - start)
        print(loss)
    """
    
    B, N, D = H.shape 
    loss = torch.zeros(1, device = H.device)
    abs_H = torch.abs(H)

    # Compute pairwise dot product interactions across all rows of V
    # v_interaction(i, j) = \Sum_{l=1}^I |V[i, l] * V[j, l]| (i.e., the influence of the components of the ith through jth dimensions of H interact through V)
    # Thus v(i, j) represents the total interaction between two dimensions i and j in H
    v_interaction_matrix = torch.matmul(V.abs(), V.abs().T)  # D x D

    # Initialize tensors to accumulate numerator and denominator across all batches and tokens
    weighted_interaction_total = torch.zeros((B, N), device = H.device)
    sum_interaction_total = torch.zeros((B, N), device = H.device)

    # Compute the numerator (distance-weighted) and denominator (unweighted) on the fly
    for i in target_dims:
        for j in range(D):
            if i != j: 
                # Compute interaction term |h_i * h_j * v_i * v_j| on the fly
                interaction = abs_H[:, :, i] * abs_H[:, :, j]

                # Use precomputed pairwise element-wise interaction in V
                v_interaction = v_interaction_matrix[i, j]
                
                # Calculate distance-weighted interaction for numerator
                weighted_interaction = interaction * v_interaction * abs(i - j)

                # Accumulate weighted interactions (numerator) across batches and tokens
                weighted_interaction_total += weighted_interaction

                # Accumulate unweighted interactions (denominator)
                sum_interaction_total += interaction * v_interaction

    loss = weighted_interaction_total.sum()
    sum_interactions = sum_interaction_total.sum()

    # Normalize the loss
    if sum_interactions > 0:
        loss = loss / sum_interactions

    return loss

def get_modularity_loss_v4(H, V, target_dims):
    """
    Computes the normalized L1 distance-weighted modularity loss for the given input tensors H and V,
     considering only interactions involving the specified target dimensions across multiple layers.

    Details:
        The same as `get_modularity_loss_v3`, but with two changes:
            1. Accepts an H tensor of B x K x N x D, instead of B x N x D, and vectorizes over K layers
            2. Calculates normalization seperately per target dimension and per layer, instead of a shared normalization for each.

    Params:
        @H: A B x K x N x D hidden state tensor
        @V: The D x I weight tensor which is multiplied by H
        @target_dims: The list of dimensions (0-indexed) within H to consider for the restricted loss calculation.

    Returns:
        A K x ||target dimensions|| tensor of L1 modularity losses.

    Examples:
        torch.manual_seed(0)
        H = torch.randn(10, 1024, 3072, dtype = torch.bfloat16, device = 'cuda').unsqueeze(1).repeat(1, 32, 1, 1)
        V = torch.randn(3072, 8192, dtype = torch.bfloat16, device = 'cuda')
        start = time.time()
        loss = get_modularity_loss_v4(H, V, target_dims = [3040, 3052])
        end = time.time()
        print(end - start)
        print(loss)
    """
    
    B, K, N, D = H.shape 
    abs_H = torch.abs(H)

    # Precompute pairwise interaction between dimensions in V (D x D matrix)
    v_interaction_matrix = torch.matmul(V.abs(), V.abs().T)  # Shape: D x D

    # Initialize the K x ||target dimensions|| output tensor for storing losses for each layer and target dimension
    losses = torch.zeros((K, len(target_dims)), device = H.device)

    # Compute the numerator (distance-weighted) and denominator (unweighted) for each target dimension
    for idx, i in enumerate(target_dims):
        # Initialize accumulators for the current target dimension
        weighted_interaction_total = torch.zeros((B, K, N), device=H.device)
        sum_interaction_total = torch.zeros((B, K, N), device=H.device)
        
        for j in range(D):
            if i != j: 
                # Compute interaction term |h_i * h_j * v_i * v_j| across B x K x N on the fly
                interaction = abs_H[..., i] * abs_H[..., j]  # Shape: B x K x N

                # Use precomputed pairwise element-wise interaction in V
                v_interaction = v_interaction_matrix[i, j]
                
                # Calculate distance-weighted interaction for numerator
                weighted_interaction = interaction * v_interaction * abs(i - j)

                # Accumulate weighted interactions (numerator) across batches, layers, and tokens
                weighted_interaction_total += weighted_interaction

                # Accumulate unweighted interactions (denominator)
                sum_interaction_total += interaction * v_interaction

        # Sum over batches and tokens to compute the loss for each layer for the current target dimension
        layer_weighted_sums = weighted_interaction_total.sum(dim = (0, 2))  # Shape: K
        layer_sum_interactions = sum_interaction_total.sum(dim = (0, 2))    # Shape: K
    
        # Normalize each layer's loss for the current target dimension
        layer_losses = torch.where(
            layer_sum_interactions > 1e-8, 
            layer_weighted_sums / (layer_sum_interactions + 1e-8), 
            torch.zeros_like(layer_sum_interactions)
            )
        
        # Store the result in the losses tensor
        losses[:, idx] = layer_losses

    return losses

# def get_modularity_loss_cubed(H, V, target_dims, neighbor_range=1):
# TBD: Same as v3 but only calculates loss within a certian neighbor range (e.g. if neighbor_range = 3072, the result equals v2 result)
#     """
#     Computes the normalized L1 distance-weighted modularity loss for the given input tensors H and V,
#     considering only interactions involving the specified target dimensions, while avoiding memory issues.
    
#     Params:
#         @H: A B x N x D hidden state tensor
#         @V: The D x I weight tensor which is multiplied by H
#         @target_dims: The list of dimensions (0-indexed) within H to consider for the restricted loss calculation.

#     Returns:
#         The computed normalized L1 modularity loss for the restricted dimensions.
#     """
    
#     B, N, D = H.shape  # Batch size (B), token length (N), dimension (D)

#     # Precompute the absolute values of H for target dimensions
#     abs_H = torch.abs(H)

#     # Compute pairwise element-wise interactions across all rows of V
#     v_interaction_matrix = torch.matmul(V.abs(), V.abs().T)  # D x D

#     # Initialize tensors to accumulate numerator and denominator across all batches and tokens
#     weighted_interaction_total = torch.zeros((B, N), device=H.device)
#     sum_interaction_total = torch.zeros((B, N), device=H.device)

#     # Compute the numerator (distance-weighted) and denominator (unweighted)
#     for i in target_dims:
#         # Only consider neighbors within the specified range
#         for j in range(max(0, i - neighbor_range), min(D, i + neighbor_range + 1)):
#             if i != j:  # Skip self-interaction
#                 # Compute the pre-cached interaction term |h_i * h_j| using the precomputed absolute values
#                 interaction = abs_H[:, :, i] * abs_H[:, :, j]

#                 # Use precomputed pairwise element-wise interaction in V
#                 v_interaction = v_interaction_matrix[i, j]
                
#                 # Calculate distance-weighted interaction for numerator
#                 weighted_interaction = interaction * v_interaction * abs(i - j)

#                 # Accumulate weighted interactions (numerator) across batches and tokens
#                 weighted_interaction_total += weighted_interaction

#                 # Accumulate unweighted interactions (denominator)
#                 sum_interaction_total += interaction * v_interaction

#     # Sum over batches and tokens outside the loop to compute the final loss
#     loss = weighted_interaction_total.sum()
#     sum_interactions = sum_interaction_total.sum()

#     # Normalize the loss
#     if sum_interactions > 0:
#         loss = loss / sum_interactions

#     return loss
