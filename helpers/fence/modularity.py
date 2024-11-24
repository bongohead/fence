# Modularity loss functions
 
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

def get_modularity_loss_v5(H, V, target_dims, block_size=1):
    """
    Computes the normalized L1 distance-weighted modularity loss for the given input tensors H and V,
     considering only interactions involving the specified target dimensions across multiple layers.

    Details:
        The same as `get_modularity_loss_v4`, but with two changes:
            1. Vectorizes over dimensions
            2. Allows for grouping dimensions into "blocks", where elements in each block are considered the same distance. Set block_size = 1 to replicate `get_modularity_loss_v4`. 

        Reduced benchmark time with Df = 3 from .29s (v4) to .02s (v5)

    Params:
        @H: A B x K x N x D hidden state tensor
        @V: The D x I weight tensor which is multiplied by H
        @target_dims: The list of dimensions (0-indexed) within H to consider for the restricted loss calculation.
        @block_size: The size of each block, s.t. elements of each block are considered the same distance away from the target dim. Set block_size = 1 to replicate `get_modularity_loss_v4`.
         With block_size=10: distances 0-9 get weight 5; distances 10-19 get weight 15; distances 20-29 get weight 25; etc.


    Returns:
        A K x ||target dimensions|| tensor of L1 modularity losses.

    Examples:
        torch.manual_seed(0)
        H = torch.randn(10, 1024, 3072, dtype = torch.bfloat16, device = 'cuda').unsqueeze(1).repeat(1, 32, 1, 1)
        V = torch.randn(3072, 8192, dtype = torch.bfloat16, device = 'cuda')
        start = time.time()
        loss = get_modularity_loss_v5(H, V, target_dims = [3040, 3052], block_size = 1)
        end = time.time()
        print(end - start)
        print(loss)
    """ 
        
    B, K, N, D = H.shape
    losses = torch.zeros((K, len(target_dims)), device = H.device)
    abs_H = torch.abs(H)

    # Compute pairwise dot product interactions across all rows of V
    # v_interaction(i, j) = \Sum_{l=1}^I |V[i, l] * V[j, l]| (i.e., the influence of the components of the ith through jth dimensions of H interact through V)
    # Thus v(i, j) represents the total interaction between two dimensions i and j in H
    v_interaction_matrix = torch.matmul(V.abs(), V.abs().T)  # D x D

    # Cast distances into blocks
    distances = torch.arange(D, device = H.device)
    blocked_distances = torch.zeros((len(target_dims), D), device = H.device)
    
    for idx, target_dim in enumerate(target_dims):
        raw_distances = torch.abs(distances - target_dim)
        blocked_idx = torch.div(raw_distances, block_size, rounding_mode='floor')
        blocked_distances[idx] = blocked_idx * block_size
    
    # Iterate through target dims
    for idx, i in enumerate(target_dims):
        block_weights = blocked_distances[idx]
        
        # Create mask to exclude self-interaction
        mask = torch.ones(D, device = H.device)
        mask[i] = 0
        
        interactions = abs_H[..., i].unsqueeze(-1) * abs_H
        v_interactions = v_interaction_matrix[i] * mask  # Apply mask to exclude self-interaction, to mimic v4
        
        weighted_sum = (interactions * v_interactions * block_weights).sum(dim=-1)
        total_sum = (interactions * v_interactions).sum(dim=-1)
        
        # Match original zero handling
        losses[:, idx] = torch.where(
            total_sum.sum(dim = (0, 2)) > 0,
            weighted_sum.sum(dim = (0, 2)) / (total_sum.sum(dim = (0, 2)) + 1e-8),
            torch.zeros_like(total_sum.sum(dim = (0, 2)))
        )

    return losses

def get_modularity_loss_v6(H, V, target_dims, block_size = 1, chunk_size = 256):
    """
    Computes the normalized L1 distance-weighted modularity loss for the given input tensors H and V,
     considering only interactions involving the specified target dimensions across multiple layers.

    Details:
        The same as `get_modularity_loss_v5`, but with several changes for memory efficiency:
            1. Chunk-wise processing; reducing the H interaction matrix of size B x K x N x D into pieces of size B x K x N x chunk_size
            2. Only calculates needed rows for the V interaction matrix instead of the entire D x D matrix

        Reduced peak memory excl inputs in below example from 12GB -> 1GB (note that `get_modularity_loss_v4` used ~2GB)

    Params:
        @H: A B x K x N x D hidden state tensor
        @V: The D x I weight tensor which is multiplied by H
        @target_dims: The list of dimensions (0-indexed) within H to consider for the restricted loss calculation.
        @block_size: The size of each block, s.t. elements of each block are considered the same distance away from the target dim. Set block_size = 1 to replicate `get_modularity_loss_v4`.
         With block_size=10: distances 0-9 get weight 5; distances 10-19 get weight 15; distances 20-29 get weight 25; etc.
        @chunk_size: The chunk size when calculating the H interaction matrix; toggle this appropriately for memory tradeoff (larger size = more memory)

    Returns:
        A K x ||target dimensions|| tensor of L1 modularity losses.

    Examples:
        torch.manual_seed(0)
        H = torch.randn(10, 1024, 3072, dtype = torch.bfloat16, device = 'cuda').unsqueeze(1).repeat(1, 32, 1, 1)
        V = torch.randn(3072, 8192, dtype = torch.bfloat16, device = 'cuda')
        start = time.time()
        loss = get_modularity_loss_v6(H, V, target_dims = [3040, 3052], block_size = 1, chunk_size = 256)
        end = time.time()
        print(end - start)
        print(loss)
    """ 

    B, K, N, D = H.shape
    losses = torch.zeros((K, len(target_dims)), device=H.device)
    
    # Compute pairwise dot product interactions across all rows of V
    # v_interaction(i, j) = \Sum_{l=1}^I |V[i, l] * V[j, l]| (i.e., the influence of the components of the ith through jth dimensions of H interact through V)
    # Thus v(i, j) represents the total interaction between two dimensions i and j in H   
    # Note that for v6, this only computes needed rows; v5 computed the full D x D interaction matrix: O(||target_dims|| x D) instead of O(D^2)
    target_rows = torch.tensor(target_dims, device = H.device)
    v_target = V.abs()[target_rows]  # len(target_dims) x I
    v_interaction_rows = torch.matmul(v_target, V.abs().T)  # len(target_dims) x D
    
    # Compute distances once, with proper scaling
    distances = torch.arange(D, device = H.device)
    blocked_distances = torch.zeros((len(target_dims), D), device = H.device)
    
    for idx, target_dim in enumerate(target_dims):        
        raw_distances = torch.abs(distances - target_dim)
        blocked_idx = torch.div(raw_distances, block_size, rounding_mode = 'floor')
        blocked_distances[idx] = blocked_idx * block_size
    
    # Now iterate over the target dims and calculate losses
    for idx, i in enumerate(target_dims):
        h_i = H[..., i].abs().unsqueeze(-1)  # B x K x N x 1
        v_interactions = v_interaction_rows[idx] # D
        
        # Mask self-interaction
        mask = torch.ones(D, device=H.device)
        mask[i] = 0
        v_interactions = v_interactions * mask
        
        # Use in-place operations where possible
        weighted_sum = torch.zeros((B, K, N), device = H.device)
        total_sum = torch.zeros((B, K, N), device = H.device)
        
        # Process in chunks for memory efficiency: O(BKND) → O(BKND/num_chunks)
        for chunk_start in range(0, D, chunk_size):
            chunk_end = min(chunk_start + chunk_size, D)
            
            # Process chunk
            h_chunk = H[..., chunk_start:chunk_end].abs()
            chunk_interaction = (h_i * h_chunk * v_interactions[chunk_start:chunk_end])
            # Uses in-place addition for memory efficiency
            weighted_sum.add_((chunk_interaction * blocked_distances[idx, chunk_start:chunk_end]).sum(dim = -1))
            total_sum.add_(chunk_interaction.sum(dim = -1))
        
        # Compute normalized loss
        losses[:, idx] = torch.where(
            total_sum.sum(dim = (0, 2)) > 0,
            weighted_sum.sum(dim = (0, 2)) / (total_sum.sum(dim = (0, 2)) + 1e-8),
            torch.zeros_like(total_sum.sum(dim = (0, 2)))
        )
    
    return losses
