import json
import hashlib
from datetime import datetime

class ImmutableAuditLogger:
    """Logs system events, proofs, and compliance traces (Section 10)."""
    def __init__(self, log_path: str = "audit_log.jsonl"):
        self.log_path = log_path
        self._prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    def _compute_hash(self, payload: dict) -> str:
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256((self._prev_hash + payload_str).encode('utf-8')).hexdigest()

    def log_event(self, event_type: str, operator_id: str, data: dict):
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        payload = {
            "timestamp": timestamp,
            "event_type": event_type,
            "operator_id": operator_id,
            "data": data
        }
        
        current_hash = self._compute_hash(payload)
        
        log_entry = {
            "prev_hash": self._prev_hash,
            "hash": current_hash,
            "payload": payload
        }
        
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        self._prev_hash = current_hash
