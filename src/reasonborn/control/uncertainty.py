import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyEstimator:
    """Calculates Model Confidence using Temperature Scaling and MC-Dropout."""
    def __init__(self, model, num_mc_samples: int = 5):
        self.model = model
        self.num_mc_samples = num_mc_samples
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Learned during calibration

    def calibrate(self, valid_loader, optimizer):
        """Optimizes temperature T on a validation set to minimize NLL."""
        self.temperature.requires_grad = True
        for batch in valid_loader:
            logits = self.model(batch['input_ids']).logits
            loss = F.cross_entropy(logits / self.temperature, batch['labels'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.temperature.requires_grad = False

    def estimate_confidence(self, input_ids: torch.Tensor) -> float:
        """Runs MC-Dropout to estimate epistemic uncertainty."""
        self.model.train() # Enable dropout
        probs = []
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                logits = self.model(input_ids).logits
                scaled_logits = logits / self.temperature
                probs.append(F.softmax(scaled_logits, dim=-1))
                
        # Calculate variance of predictions across dropout passes
        prob_tensor = torch.stack(probs)
        mean_prob = prob_tensor.mean(dim=0)
        variance = prob_tensor.var(dim=0).mean().item()
        
        # High variance -> Low confidence
        confidence = 1.0 - min(1.0, variance * 10.0) 
        self.model.eval()
        return max(0.0, confidence)
