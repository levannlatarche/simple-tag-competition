"""
Template for student agent submission.

Students should implement the StudentAgent class for the predator only.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class StudentAgent:
    """
    Template agent class for Simple Tag competition.
    
    Students must implement this class with their own agent logic.
    The agent should handle only the "predator" type. The prey is provided publicly by the course.
    """
    
    def __init__(self):
        """
        Initialize your predator agent.
        """
        # Example: Load your trained models
        # Get the directory where this file is located
        self.submission_dir = Path(__file__).parent
        self.obs_dim=14
        self.action_dim=5
        # Example: Load predator model
        model_path = self.submission_dir / "predator_model.pth"
        self.policy = PPOActorCritic(input_dim=self.obs_dim, output_dim=self.action_dim)
        if model_path.exists():
            self.policy.load_state_dict(torch.load(model_path))
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad = False
        pass
    
    def get_action(self, observation, agent_id: str):
        """
        Get action for the given observation.
        
        Args:
            observation: Agent's observation from the environment (numpy array)
                         - Predator (adversary): shape (14,)
            agent_id (str): Unique identifier for this agent instance
            
        Returns:
            action: Discrete action in range [0, 4]
                    0 = no action
                    1 = move left
                    2 = move right  
                    3 = move down
                    4 = move up
        """
        # IMPLEMENT YOUR POLICY HERE
        
        # Example random policy (replace with your trained policy):
        # Action space is Discrete(5) by default
        # Note: During evaluation, RNGs are seeded per episode for determinism
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # shape (1, 14)
        with torch.no_grad():  # no gradients in evaluation
            logits = self.policy(obs)        # shape (1, 5)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
        
        return int(action.item())
    
    def load_model(self, model_path):
        """
        Helper method to load a PyTorch model.
        
        Args:
            model_path: Path to the .pth file
            
        Returns:
            Loaded model
        """
        # Example implementation:
        model = PPOActorCritic()
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        return model
        pass


# Example Neural Network Architecture (customize as needed)
class PPOActorCritic(nn.Module):
    """
    Standard PPO network with shared encoder + policy head + value head.
    Recommended architecture for MPE Simple Tag.
    """
    def __init__(self, input_dim=14, output_dim=5, hidden=128):
        super().__init__()

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.policy_head = nn.Linear(hidden, input_dim)

        # Value head (critic)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        Returns (logits, value)
        """
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

    def get_action_logits(self, x):
        """Convenience function for evaluation."""
        h = self.shared(x)
        return self.policy_head(h)

if __name__ == "__main__":
    # Example usage
    print("Testing StudentAgent...")
    
    # Test predator agent (adversary has 14-dim observation)
    predator_agent = StudentAgent()
    predator_obs = np.random.randn(14)  # Predator observation size
    predator_action = predator_agent.get_action(predator_obs, "adversary_0")
    print(f"Predator observation shape: {predator_obs.shape}")
    print(f"Predator action: {predator_action} (should be in [0, 4])")
    
    print("✓ Agent template is working!")
