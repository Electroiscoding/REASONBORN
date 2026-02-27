import torch
from reasonborn.architecture.hybrid_attention import HybridAttentionLayer

def test_hybrid_attention_forward():
    class MockConfig:
        d_model = 128
        num_heads = 4
        
    layer = HybridAttentionLayer(MockConfig())
    
    # Batch=2, SeqLen=1024, Dim=128
    dummy_input = torch.randn(2, 1024, 128)
    output = layer(dummy_input)
    
    assert output.shape == dummy_input.shape
    assert not torch.isnan(output).any()
