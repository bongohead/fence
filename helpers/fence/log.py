import torch 
import numpy as np 

@torch.no_grad()
def get_gradient_stats(model):
    """
    Returns a dictionary of gradient statistics.

    Params:
    model: The model
    """
    stats = {}
    grad_norms = {}
    grad_maxs = {}

    VANISHING_THRESHOLD = 1e-4
    EXPLODING_THRESHOLD = 10.0 
    num_params = 0
    num_vanishing = 0
    num_exploding = 0
    
    # Track transformer layer gradients separately
    layer_stats = {}

    # Compute stats for each parameter
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach()
            
            # Basic statistics
            grad_norm = grad.norm().item()
            grad_max = grad.abs().max().item()
            grad_norms[name] = grad_norm
            grad_maxs[name] = grad_max
            
            # Count vanishing/exploding
            num_params += 1
            if grad_norm < VANISHING_THRESHOLD:
                num_vanishing += 1
            if grad_norm > EXPLODING_THRESHOLD:
                num_exploding += 1
                
            # Collect per-layer statistics
            # Extract layer number for transformer layers (assumes names like "layers.0", "layers.1", etc)
            if 'layers.' in name:
                layer_num = str(int(name.split('layers.')[1].split('.')[0]) + 1)  # Convert to 1-based
                if layer_num not in layer_stats:
                    layer_stats[layer_num] = []
                layer_stats[layer_num].append(grad_norm)

    # Aggregate statistics
    stats.update({
        'max_l2_norm': max(grad_norms.values()),
        'mean_l2_norm': sum(grad_norms.values()) / len(grad_norms),
        'max_abs_value': max(grad_maxs.values()),
        'has_nan': any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None),
        'has_inf': any(torch.isinf(p.grad).any().item() for p in model.parameters() if p.grad is not None),
        'pct_vanishing': (num_vanishing / num_params) * 100,
        'pct_exploding': (num_exploding / num_params) * 100,
        'max_l2_norm_by_layer': {},
        'mean_l2_norm_by_layer': {}
    })
    
    # Add per-transformer-layer statistics
    for layer_num, layer_grads in layer_stats.items():
        stats['max_l2_norm_by_layer'][f'l{layer_num}'] = np.max(layer_grads)
        stats['mean_l2_norm_by_layer'][f'l{layer_num}']  = np.mean(layer_grads)

    return stats
