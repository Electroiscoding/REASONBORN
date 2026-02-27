from reasonborn.reasoning.engine import Node, NestedCoTEngine

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
    engine = NestedCoTEngine(MockModel(), max_depth=3)
    # Mocking verify to pass
    engine._verify_solution = lambda n, s: {"passed": True, "confidence": 1.0, "proof": {}}
    
    final, _ = engine.run("solve this complex problem", {})
    assert final == "synthesized"
