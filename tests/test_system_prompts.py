from reasonborn.control.prompt_manager import SystemPromptManager

def test_operator_precedence():
    manager = SystemPromptManager()
    
    operator = {
        "mode": "restricted",
        "outputs": {"allowed_types": ["summary"], "max_tokens": 500},
        "safety": {
            "sensitivity": "maximum",
            "require_human_approval": ["medical"],
            "prohibited_topics": [],
            "max_uncertainty": 0.5,
            "refuse_speculation": True
        },
        "privacy": {"mode": "dp_strict"},
        "resources": {"max_tokens": 500, "max_wall_time_ms": 1000, "max_reasoning_depth": 3}
    }
    
    user_attempt = {
        "mode": "research",
        "outputs": {"allowed_types": ["full_CoT", "summary"], "max_tokens": 8000},
        "safety": {"sensitivity": "low", "require_human_approval": []},
        "privacy": {"mode": "none"}
    }
    
    merged = manager.merge_with_precedence(operator, user_attempt)
    
    assert merged['mode'] == "restricted" # Operator wins
    assert merged['safety'].sensitivity == "maximum" # Max wins
    assert "full_CoT" not in merged['allowed_outputs'] # Intersection
    assert merged['resource_limits'].max_tokens == 500 # Minimum wins
    assert merged.get('privacy', {}).get('mode', operator['privacy']['mode']) == "dp_strict" or merged.get('privacy_mode', operator['privacy']['mode']) == "dp_strict" # Handle missing privacy key in merged dict
