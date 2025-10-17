import torch

def extract_mlp_weights_and_gradients(train_pipeline):
    """
    Extract weights and gradients from all Linear layers in the MLP components in a DLRM model.
    
    Args:
        train_pipeline: TrainPipelineSparseDist_Log object
        
    Returns:
        dict: Dictionary containing weights and gradients for each MLP layer
    """
    model = train_pipeline._model
    
    # Navigate to the actual DLRM model through the wrappers
    # DistributedModelParallel -> DistributedDataParallel -> DLRMTrain -> DLRM_DCN
    dlrm_model = model.module.model
    
    mlp_data = {
        'dense_arch': {},
        'over_arch': {},
        'crossnet': {}
    }
    
    # Extract dense_arch MLP layers (bottom MLP)
    dense_mlp = dlrm_model.dense_arch.model._mlp
    for i, layer in enumerate(dense_mlp):
        if hasattr(layer, '_linear'):
            linear = layer._linear
            mlp_data['dense_arch'][f'layer_{i}'] = {
                'weight': linear.weight.data.clone(),
                'bias': linear.bias.data.clone() if linear.bias is not None else None,
                'weight_grad': linear.weight.grad.clone() if linear.weight.grad is not None else None,
                'bias_grad': linear.bias.grad.clone() if linear.bias is not None and linear.bias.grad is not None else None,
                'shape': linear.weight.shape
            }
    
    # Extract over_arch MLP layers (top MLP)
    over_arch_sequential = dlrm_model.over_arch.model
    
    # First part: MLP layers within over_arch
    if len(over_arch_sequential) > 0 and hasattr(over_arch_sequential[0], '_mlp'):
        over_mlp = over_arch_sequential[0]._mlp
        for i, layer in enumerate(over_mlp):
            if hasattr(layer, '_linear'):
                linear = layer._linear
                mlp_data['over_arch'][f'mlp_layer_{i}'] = {
                    'weight': linear.weight.data.clone(),
                    'bias': linear.bias.data.clone() if linear.bias is not None else None,
                    'weight_grad': linear.weight.grad.clone() if linear.weight.grad is not None else None,
                    'bias_grad': linear.bias.grad.clone() if linear.bias is not None and linear.bias.grad is not None else None,
                    'shape': linear.weight.shape
                }
    
    # Second part: Final Linear layer in over_arch
    if len(over_arch_sequential) > 1 and isinstance(over_arch_sequential[1], torch.nn.Linear):
        final_linear = over_arch_sequential[1]
        mlp_data['over_arch']['final_linear'] = {
            'weight': final_linear.weight.data.clone(),
            'bias': final_linear.bias.data.clone() if final_linear.bias is not None else None,
            'weight_grad': final_linear.weight.grad.clone() if final_linear.weight.grad is not None else None,
            'bias_grad': final_linear.bias.grad.clone() if final_linear.bias is not None and final_linear.bias.grad is not None else None,
            'shape': final_linear.weight.shape,
        }
    
    # Extract DCN crossnet parameters (also linear transformations)
    if hasattr(dlrm_model.inter_arch, 'crossnet'):
        crossnet = dlrm_model.inter_arch.crossnet
        
        # W_kernels
        for i, w_kernel in enumerate(crossnet.W_kernels):
            mlp_data['crossnet'][f'W_kernel_{i}'] = {
                'weight': w_kernel.data.clone(),
                'weight_grad': w_kernel.grad.clone() if w_kernel.grad is not None else None,
                'shape': w_kernel.shape
            }
        
        # V_kernels
        for i, v_kernel in enumerate(crossnet.V_kernels):
            mlp_data['crossnet'][f'V_kernel_{i}'] = {
                'weight': v_kernel.data.clone(),
                'weight_grad': v_kernel.grad.clone() if v_kernel.grad is not None else None,
                'shape': v_kernel.shape
            }
        
        # Bias parameters
        for i, bias in enumerate(crossnet.bias):
            mlp_data['crossnet'][f'bias_{i}'] = {
                'bias': bias.data.clone(),
                'bias_grad': bias.grad.clone() if bias.grad is not None else None,
                'shape': bias.shape
            }
    
    return mlp_data


def print_mlp_summary(mlp_data):
    """Print a summary of the extracted MLP data."""

    print(f"=== MLP Weights and Gradients Summary ===")
    
    for arch_name, arch_data in mlp_data.items():
        print(f"\n{arch_name.upper()}:")
        for layer_name, layer_data in arch_data.items():
            print(f"  {layer_name}:")
            
            # Handle weight parameters
            if 'weight' in layer_data:
                weight = layer_data['weight']
                weight_grad = layer_data['weight_grad']
                print(f"    Weight: {weight.shape} on {weight.device}")
                print(f"    Weight mean: {weight.mean().item():.6e}, std: {weight.std().item():.6e}")
                print(f"    Weight max: {weight.max().item():.6e}")
                
                if weight_grad is not None:
                    print(f"    Weight grad max: {weight_grad.max().item():.6e}")
                    print(f"    Weight grad mean: {weight_grad.mean().item():.6e}")
                else:
                    print(f"    Weight grad: None")
            
            # Handle bias parameters
            if 'bias' in layer_data and layer_data['bias'] is not None:
                bias = layer_data['bias']
                bias_grad = layer_data['bias_grad']
                print(f"    Bias: {bias.shape} on {bias.device}")
                print(f"    Bias mean: {bias.mean().item():.6e}, std: {bias.std().item():.6e}")
                
                if bias_grad is not None:
                    print(f"    Bias grad max: {bias_grad.max().item():.6e}")
                else:
                    print(f"    Bias grad: None")
            
            # Add layer dimension info for Linear layers
            if 'in_features' in layer_data:
                print(f"    Dimensions: {layer_data['in_features']} -> {layer_data['out_features']}")


def save_mlp_data(mlp_data, filename):
    """Save MLP data to file."""
    
    save_data = {
        'mlp_data': mlp_data
    }
    torch.save(save_data, filename)
    print(f"MLP data saved to {filename}")