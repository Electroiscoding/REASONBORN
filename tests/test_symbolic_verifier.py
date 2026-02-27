from reasonborn.reasoning.verification.symbolic import SymbolicVerifier

def test_z3_symbolic_solver():
    verifier = SymbolicVerifier()
    
    goal = "solve for x and y"
    # Provide a text string that satisfies basic regex extraction (x=5, y=10)
    proposed_solution = "Therefore x = 5.0 and y = 10.0"
    
    assert verifier.is_applicable("prove that x=5") == True
    
    result = verifier.verify(goal, proposed_solution)
    assert result["passed"] == True
    assert result["confidence"] == 1.0
