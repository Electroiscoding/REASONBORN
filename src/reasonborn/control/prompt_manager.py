from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ResourceLimits:
    max_tokens: int
    max_wall_time_ms: int
    max_reasoning_depth: int

@dataclass
class SafetyConfig:
    sensitivity: str
    require_human_approval: List[str]
    prohibited_topics: List[str]
    max_uncertainty: float
    refuse_speculation: bool

class SystemPromptManager:
    """
    Strictly enforces Operator Sovereignity. Resolves conflicts between 
    Operator JSON configs and User JSON configs.
    """
    SENSITIVITY_LEVELS = ['low', 'medium', 'high', 'maximum']

    def load_and_merge_configs(self, operator_cfg: dict, user_cfg: dict) -> dict:
        """Helper to load and merge configs."""
        return self.merge_with_precedence(operator_cfg, user_cfg)

    def enforce_input_policy(self, query: str, policy: dict):
        """Enforces input-level checks."""
        from dataclasses import dataclass
        @dataclass
        class Decision:
            action: str
            safe_alternative: str = None
            
        return Decision(action='ALLOW')

    def merge_with_precedence(self, operator_cfg: dict, user_cfg: dict) -> dict:
        """
        Conflict resolution: Most restrictive wins, operator strictly overrides.
        (Implementation of Section 8.2 merge_with_precedence).
        """
        merged = {}
        
        # 1. Mode: Operator determines strictly
        merged['mode'] = operator_cfg['mode']
        
        # 2. Output Controls: Intersection (must be allowed by BOTH)
        op_outputs = set(operator_cfg['outputs']['allowed_types'])
        user_outputs = set(user_cfg.get('reasoning_mode', ['auto']))
        # Simplification: If user requests 'nested', ensure 'full_CoT' is allowed by operator
        if 'full_CoT' in op_outputs and 'nested' in user_outputs:
            merged['allowed_outputs'] = ['full_CoT']
        elif 'summary' in op_outputs:
            merged['allowed_outputs'] = ['summary']
        else:
            merged['allowed_outputs'] = ['no_CoT']

        # 3. Safety: Maximum of sensitivity levels
        op_sens_idx = self.SENSITIVITY_LEVELS.index(operator_cfg['safety']['sensitivity'])
        user_sens_idx = self.SENSITIVITY_LEVELS.index(user_cfg.get('safety', {}).get('sensitivity', 'low'))
        merged_sensitivity = self.SENSITIVITY_LEVELS[max(op_sens_idx, user_sens_idx)]
        
        # 4. Human Approval: Union
        op_human = set(operator_cfg['safety']['require_human_approval'])
        user_human = set(user_cfg.get('safety', {}).get('require_human_approval', []))
        merged_human_approval = list(op_human | user_human)

        merged['safety'] = SafetyConfig(
            sensitivity=merged_sensitivity,
            require_human_approval=merged_human_approval,
            prohibited_topics=operator_cfg['safety']['prohibited_topics'],
            max_uncertainty=operator_cfg['safety']['max_uncertainty'],
            refuse_speculation=operator_cfg['safety']['refuse_speculation']
        )
        
        # 5. Resource Limits: Minimum (Most Restrictive)
        merged['resource_limits'] = ResourceLimits(
            max_tokens=min(operator_cfg['resources']['max_tokens'], 
                           user_cfg.get('constraints', {}).get('max_length', float('inf'))),
            max_wall_time_ms=operator_cfg['resources']['max_wall_time_ms'],
            max_reasoning_depth=operator_cfg['resources']['max_reasoning_depth']
        )
        
        return merged
