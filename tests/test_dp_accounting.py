import pytest
from reasonborn.privacy.privacy_accountant import RenyiPrivacyAccountant

def test_renyi_accountant():
    accountant = RenyiPrivacyAccountant(target_delta=1e-5)
    
    # Simulate 100 training steps
    for _ in range(100):
        accountant.record_update(noise_multiplier=1.1, sample_rate=0.01, steps=1)
        
    epsilon = accountant.get_current_epsilon()
    assert epsilon > 0.0
    
    # Should flag if budget exceeds an arbitrarily low threshold
    assert accountant.check_budget_exceeded(max_epsilon=0.001) == True
