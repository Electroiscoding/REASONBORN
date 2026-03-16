import z3
import re
import sympy as sp
from typing import Dict, Any, List, Optional, Union

class SymbolicVerifier:
    """
    Module [3]: Neuro-symbolic verification interface.
    Implements formal SMT-LIB specification with multi-solver support.
    """

    def __init__(self, solver_type: str = "z3", timeout: int = 5):
        self.solver_type = solver_type
        self.timeout = timeout
        
        # Multi-solver support
        if solver_type == "z3":
            self.solver = z3.Solver()
            self.solver.set("timeout", timeout * 1000)  # Z3 uses milliseconds
        elif solver_type == "cvc5":
            try:
                import cvc5
                self.solver = cvc5.Solver()
                self.solver.set("timeout", timeout)
            except ImportError:
                self.solver = z3.Solver()  # Fallback
        else:
            self.solver = z3.Solver()
            self.solver.set("timeout", timeout * 1000)
        
        # Comprehensive constraint patterns
        self.constraint_patterns = {
            'equality': re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9.-]+|"[^"]*")'),
            'inequality': re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>)\s*([0-9.-]+)'),
            'logical': re.compile(r'(not|and|or|implies|iff|forall|exists)\s+([A-Za-z_][A-Za-z0-9_]*)'),
            'arithmetic': re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*(\+|-|\*|/)\s*([0-9.-]+)'),
            'function': re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([^)]+)\s*\)'),
        }
        
        # Formal proof templates
        self.proof_template = {
            'variables': [],
            'constraints': [],
            'assumptions': [],
            'conclusion': None
        }

    def is_applicable(self, goal: str) -> bool:
        """
        Determines if the subgoal contains formal mathematical/logic constraints.
        Enhanced pattern recognition for broader applicability.
        """
        goal_lower = goal.lower()
        
        # Check for formal indicators
        formal_indicators = [
            'prove', 'show', 'demonstrate', 'verify', 'validate',
            'satisfiable', 'consistent', 'theorem', 'lemma', 'corollary',
            'equation', 'inequality', 'optimization', 'constraint'
        ]
        
        # Check constraint patterns
        for pattern_type, pattern in self.constraint_patterns.items():
            if pattern.search(goal):
                return True
        
        # Check formal indicators
        return any(indicator in goal_lower for indicator in formal_indicators)

    def verify(self, claim: str, premises: List[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Translates natural language constraints to SMT-LIB and checks validity.
        
        Args:
            claim: Logical statement to verify (SMT-LIB or natural language)
            premises: Background facts and assumptions (required for proper verification)
            timeout: Maximum solver time in seconds (overrides default)
            
        Returns:
            VerificationResult with status ∈ {VALID, INVALID, UNKNOWN, TIMEOUT}
        """
        if premises is None:
            premises = []
        
        timeout = timeout or self.timeout
        
        # Clear previous state
        self.solver.push()
        
        try:
            # Step 1: Parse premises into formal constraints
            formal_premises = self._parse_to_smt(premises)
            
            # Step 2: Parse claim into formal constraint
            formal_claim = self._parse_to_smt([claim])[0] if self._parse_to_smt([claim]) else None
            
            if formal_claim is None:
                return {
                    "passed": False, 
                    "definitive": False, 
                    "confidence": 0.0, 
                    "feedback": "Unable to parse claim into formal constraints"
                }
            
            # Step 3: Add premises to solver
            for premise in formal_premises:
                self.solver.add(premise)
            
            # Step 4: Check validity by testing if negation of claim is satisfiable
            # If ¬claim ∧ premises is unsatisfiable, then claim is necessarily true
            negated_claim = self._negate_constraint(formal_claim)
            self.solver.add(negated_claim)
            
            result = self.solver.check()
            
            if result == z3.unsat:
                # VALID: Claim follows necessarily from premises
                proof_obj = self._extract_proof_object(premises, claim, "VALID", self.solver.proof())
                return {
                    "passed": True, 
                    "definitive": True, 
                    "confidence": 1.0, 
                    "proof": proof_obj,
                    "status": "VALID",
                    "reasoning": "Claim is logically entailed by premises"
                }
            elif result == z3.sat:
                # INVALID: Counterexample exists
                counterexample = self.solver.model()
                proof_obj = self._extract_proof_object(premises, claim, "INVALID", counterexample)
                return {
                    "passed": False, 
                    "definitive": True, 
                    "confidence": 0.0, 
                    "feedback": f"Counterexample found: {counterexample}",
                    "proof": proof_obj,
                    "status": "INVALID",
                    "counterexample": counterexample
                }
            else:
                # UNKNOWN: Solver timeout or other issues
                return {
                    "passed": False, 
                    "definitive": False, 
                    "confidence": 0.5, 
                    "feedback": f"Solver returned: {result}",
                    "status": "UNKNOWN"
                }
                
        except Exception as e:
            return {
                "passed": False, 
                "definitive": False, 
                "confidence": 0.0, 
                "feedback": f"SMT solver error: {str(e)}",
                "status": "ERROR"
            }
        finally:
            self.solver.pop()

    def _parse_to_smt(self, statements: List[str]) -> List[Any]:
        """
        Parse natural language statements into SMT-LIB constraints.
        Uses learned patterns and fallback to SymPy for symbolic parsing.
        """
        constraints = []
        
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
                
            parsed = False
            
            # Try each constraint pattern
            for pattern_type, pattern in self.constraint_patterns.items():
                match = pattern.search(stmt)
                if match:
                    constraint = self._pattern_to_smt_constraint(match, pattern_type)
                    if constraint:
                        constraints.append(constraint)
                        parsed = True
                        break
            
            # Fallback: try SymPy parsing
            if not parsed:
                try:
                    sympy_expr = sp.sympify(stmt)
                    constraint = self._sympy_to_smt(sympy_expr)
                    if constraint:
                        constraints.append(constraint)
                except:
                    # Last resort: treat as uninterpreted function
                    func_name = f"uninterpreted_{len(constraints)}"
                    constraints.append(f"(declare-fun {func_name} () Bool)")
                    constraints.append(f"(assert ({func_name}))")
        
        return constraints

    def _pattern_to_smt_constraint(self, match, pattern_type: str) -> Optional[str]:
        """Convert regex match to SMT-LIB constraint."""
        if pattern_type == 'equality':
            var, val = match.groups()
            return f"(= {var} {val})"
        elif pattern_type == 'inequality':
            var, op, val = match.groups()
            op_map = {'<': '<', '<=': '<=', '>': '>', '>=': '>='}
            return f"({op_map[op]} {var} {val})"
        elif pattern_type == 'logical':
            op, var = match.groups()
            return f"({op} {var})"
        else:
            return None

    def _sympy_to_smt(self, expr) -> Optional[str]:
        """Convert SymPy expression to SMT-LIB format."""
        try:
            # This is a simplified conversion - production would use proper SymPy->SMT translators
            if expr.is_Relational:
                left, right = expr.lhs, expr.rhs
                return f"(= {left} {right})"
            elif expr.is_Boolean:
                return f"({expr})"
            else:
                return None
        except:
            return None

    def _negate_constraint(self, constraint: Any) -> Any:
        """Negate a constraint for validity checking."""
        if isinstance(constraint, str):
            if constraint.startswith("(= "):
                # Equality: ¬(a = b) becomes (a ≠ b)
                return constraint.replace("(= ", "(not (= ")
            elif constraint.startswith("(< "):
                # Inequality: ¬(a < b) becomes (a ≥ b)
                return constraint.replace("(< ", "(>= ")
            elif constraint.startswith("(> "):
                return constraint.replace("(> ", "(<= ")
            else:
                return f"(not {constraint})"
        else:
            # For Z3 constraints, use logical negation
            return z3.Not(constraint)

    def _extract_proof_object(self, premises: List[str], claim: str, status: str, proof_data: Any) -> Dict[str, Any]:
        """
        Extract structured proof object in JSON-LD format.
        Per ReasonBorn.md Section 6.5 with W3C-compatible contexts.
        """
        from datetime import datetime, timezone
        import hashlib
        
        proof_id = hashlib.sha256(
            f"{claim}:{status}:{len(premises)}".encode()
        ).hexdigest()[:16]
        
        return {
            "@context": ["https://www.w3.org/2018/credentials/v1", "https://schema.org/"],
            "@type": ["VerifiableCredential", "ProofObject"],
            "proof_id": proof_id,
            "claim": claim,
            "premises": premises,
            "status": status,
            "proof": str(proof_data) if proof_data else "No formal proof available",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "solver": self.solver_type,
            "method": "smt_unsat_core" if status == "VALID" else "counterexample",
            "confidence": 1.0 if status == "VALID" else 0.0
        }
