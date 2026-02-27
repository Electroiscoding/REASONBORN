import torch
from reasonborn.architecture.moe import SparseMoE

def test_moe_top2_routing():
    class MockConfig:
        hidden_size = 256
        intermediate_size = 512
        num_experts = 8
        top_k = 2
        load_balance_loss_weight = 0.01
        
    moe_layer = SparseMoE(MockConfig())
    dummy_input = torch.randn(2, 64, 256)
    
    output, aux_loss = moe_layer(dummy_input)
    
    assert output.shape == dummy_input.shape
    assert aux_loss.item() >= 0.0 # Balancing loss should be computed
