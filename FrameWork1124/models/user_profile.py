from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class ProfileWeights:
    price_weight: float
    latency_weight: float
    
    def __post_init__(self):
        total = self.price_weight + self.latency_weight
        if not np.isclose(total, 1.0):
            raise ValueError("Weights must sum to 1.0")

class UserProfile:
    PROFILES = {
        'A': ProfileWeights(price_weight=0.8, latency_weight=0.2),    # Price-oriented
        'B': ProfileWeights(price_weight=0.2, latency_weight=0.8),    # Latency-oriented
        'C': ProfileWeights(price_weight=0.5, latency_weight=0.5),    # Balanced
    }
    
    def __init__(self, profile_type: str):
        if profile_type not in self.PROFILES:
            raise ValueError(f"Invalid profile type. Choose from: {list(self.PROFILES.keys())}")
        self.profile_type = profile_type
        self.weights = self.PROFILES[profile_type]
    
    def calculate_score(self, price: float, latency: float) -> float:
        """Calculate weighted score based on profile preferences"""
        normalized_price = self._normalize_price(price)
        normalized_latency = self._normalize_latency(latency)
        
        return (
            self.weights.price_weight * (1 - normalized_price) +
            self.weights.latency_weight * (1 - normalized_latency)
        )
    
    @staticmethod
    def _normalize_price(price: float, max_price: float = 1.0) -> float:
        """Normalize price to [0,1] range"""
        return min(price / max_price, 1.0)
    
    @staticmethod
    def _normalize_latency(latency: float, max_latency: float = 100.0) -> float:
        """Normalize latency to [0,1] range"""
        return min(latency / max_latency, 1.0) 