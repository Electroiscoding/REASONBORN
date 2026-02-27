from reasonborn.control.prompt_manager import SystemPromptManager, SystemPromptConfig

def test_operator_precedence():
    manager = SystemPromptManager()
    
    operator = SystemPromptConfig(
        mode="restricted",
        allowed_outputs=["summary"],
        safety_sensitivity="maximum",
        max_tokens=500,
        privacy_mode="dp_strict",
        require_human_approval=["medical"]
    )
    
    user_attempt = SystemPromptConfig(
        mode="research", # Trying to override mode
        allowed_outputs=["full_CoT", "summary"],
        safety_sensitivity="low", # Trying to lower safety
        max_tokens=8000,
        privacy_mode="none",
        require_human_approval=[]
    )
    
    merged = manager.merge_with_precedence(operator, user_attempt)
    
    assert merged.mode == "restricted" # Operator wins
    assert merged.safety_sensitivity == "maximum" # Max wins
    assert "full_CoT" not in merged.allowed_outputs # Intersection
    assert merged.max_tokens == 500 # Minimum wins
    assert merged.privacy_mode == "dp_strict"
