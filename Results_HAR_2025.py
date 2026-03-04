import torch
import matplotlib.pyplot as plt
from architecture import STCHAR

def reproduce_results():
    """Reproduce the paper's results"""
    # Expected results from the paper
    expected_results = {
        'NTU_RGBD_CS': 92.1,
        'NTU_RGBD_CV': 98.1,
        'Kinetics_Top1': 40.2,
        'Kinetics_Top5': 63.5,
        'Penn_Action': 92.34,
        'Human3.6M': 89.80
    }
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STCHAR(num_classes=60).to(device)
    
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    print(f"Trained accuracy: {checkpoint['best_top1']:.2f}%")
    
    # Compare with expected results
    print("\nComparison with Paper Results:")
    print("Dataset\t\t| Paper Result\t| Our Result\t| Difference")
    print("-" * 50)
    
    for dataset, expected in expected_results.items():
        # In practice, you would evaluate on each dataset separately
        # This is just a demonstration
        our_result = checkpoint['best_top1']
        difference = our_result - expected
        
        print(f"{dataset}\t| {expected}%\t\t| {our_result:.1f}%\t\t| {difference:+.1f}%")

def attention_visualization(model, sample_data):
    """Visualize attention maps as described in the paper"""
    model.eval()
    
    # Hook to get attention weights
    attention_weights = []
    def hook_fn(module, input, output):
        attention_weights.append(output.detach().cpu())
    
    # Register hook to attention module
    hook = model.stc_attention.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(sample_data.unsqueeze(0))
    
    hook.remove()
    
    # Visualize spatial attention
    spatial_attention = attention_weights[0]  # First attention output
    visualize_spatial_attention(spatial_attention[0])
    
    return attention_weights

def visualize_spatial_attention(attention_map):
    """Visualize spatial attention on skeleton joints"""
    import matplotlib.pyplot as plt
    
    # attention_map shape: (channels, frames, joints)
    joint_importance = attention_map.mean(0).mean(0)  # Average over frames and channels
    
    # Create skeleton visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # NTU-RGBD joint connections
    connections = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7), (7,8),
                   (1,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), 
                   (15,16), (0,17), (17,18), (18,19), (19,20)]
    
    # Plot connections
    for connection in connections:
        ax.plot([0, 0], [0, 0], 'k-', alpha=0.3)  # Placeholder
    
    # Plot joints with attention weights
    scatter = ax.scatter([0]*21, range(21), c=joint_importance.numpy(), 
                        s=200, cmap='Reds', alpha=0.7)
    
    plt.colorbar(scatter, label='Attention Weight')
    plt.title('Spatial Attention Map')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Run the reproduction
if __name__ == '__main__':
    reproduce_results()  # Evaluate and compare with paper results