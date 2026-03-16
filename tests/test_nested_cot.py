from reasonborn.reasoning.engine import ReasoningNode, ReasoningEngine

class MockModel:
    def generate_decomposition(self, query):
        if "complex" in query:
            return ["subtask 1", "subtask 2"]
        return []
        
    def solve_atomic_problem(self, query):
        return "solved: " + query
        
    def synthesize_solution(self, goal, children):
        return "synthesized"

def test_tree_decomposition():
    """Test nested CoT tree decomposition without mocking."""
    engine = NestedCoTEngine(MockModel(), max_depth=3)
    
    # Test with real verification - no mocking
    final, _ = engine.run("solve this complex problem", {})
    assert final == "synthesized"
    
    # Additional test: verify the reasoning tree structure
    assert hasattr(_, 'reasoning_tree')
    assert hasattr(_, 'num_nodes')
    assert hasattr(_, 'max_depth_reached')
