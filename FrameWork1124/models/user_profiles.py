from enum import Enum
from dataclasses import dataclass

class ProfileType(Enum):
    PROFILE_A = "A"  # 80% Price, 20% Latency
    PROFILE_B = "B"  # 20% Price, 80% Latency
    PROFILE_C = "C"  # 50% Price, 50% Latency

@dataclass
class UserProfile:
    """
    User profile class that handles risk preference between cost and latency.
    Higher cost_weight means more conservative in predicting HOT (prefers lower costs)
    Higher latency_weight means more aggressive in predicting HOT (prefers lower latency)
    """
    def __init__(self, cost_weight: int):
        """
        Initialize user profile with cost weight preference
        
        Args:
            cost_weight (int): Weight for cost preference (1-99)
                             Higher values mean more conservative in predicting HOT
        """
        if not 1 <= cost_weight <= 99:
            raise ValueError("Cost weight must be between 1 and 99")
            
        self.cost_weight = cost_weight / 100.0  # Normalize to 0-1 range
        self.latency_weight = (100 - cost_weight) / 100.0
        
        # Calculate decision threshold based on weights
        # Higher cost_weight = higher threshold = more conservative in predicting HOT
        self.decision_threshold = 0.5 + (self.cost_weight - 0.5) * 0.4
        
    def __str__(self):
        return f"UserProfile(cost_weight={self.cost_weight:.2f}, latency_weight={self.latency_weight:.2f}, decision_threshold={self.decision_threshold:.2f})" 