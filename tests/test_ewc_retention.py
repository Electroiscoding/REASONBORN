import torch
from reasonborn.learning.continual_learner import ContinualLearner

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)
        
def test_ewc_penalty_calculation():
    model = MockModel()
    
    class MockConfig:
        lambda_ewc = 1000.0
        retention_threshold = 0.95
        
    learner = ContinualLearner(model, MockConfig())
    
    # Artificial drift
    with torch.no_grad():
        model.fc.weight += 0.5
        
    # Set mock fisher diagonal to 1.0
    for name in learner.fisher_diag:
        learner.fisher_diag[name] = torch.ones_like(learner.fisher_diag[name])
        
    penalty = learner.compute_ewc_loss()
    assert penalty.item() > 0.0 # EWC should penalize the weight change
