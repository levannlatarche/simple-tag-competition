import torch
from pettingzoo.mpe import simple_tag_v3
from agent import PPOActorCritic
from prey_agent import StudentAgent

prey_agent = StudentAgent()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# prey_agent.model.to(device)

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, logprob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

def collect_trajectories_multi(env, policy, prey_agent, rollout_len=1024):
    """
    Collecte des trajectoires multi-prédateurs tout en incluant la proie
    pour que l'environnement ne renvoie pas de KeyError.
    Seuls les prédateurs sont ajoutés au buffer.

    Args:
        env: PettingZoo parallel_env
        policy: PPOActorCritic
        prey_agent: StudentAgent pour la proie
        rollout_len: nombre de steps à collecter
    """
    buffer = RolloutBuffer()
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # certaines versions renvoient (obs, infos)

    for _ in range(rollout_len):
        actions = {}
        logprobs_step = {}
        values_step = {}

        # 1) Pour chaque agent actif
        for agent_id in env.agents:
            if "adversary" in agent_id:
                # Prédateur → PPO
                obs_t = torch.tensor(obs[agent_id], dtype=torch.float32).unsqueeze(0)
                logits, value = policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

                actions[agent_id] = action.item()
                logprobs_step[agent_id] = logprob.item()
                values_step[agent_id] = value.item()
            else:
                # Proie → action fixe ou via StudentAgent
                actions[agent_id] = prey_agent.get_action(obs[agent_id], agent_id)
                # ne pas stocker dans le buffer

        # 2) Step environment avec toutes les actions
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        if isinstance(next_obs, tuple):
            next_obs = next_obs[0]

        # 3) Stocker uniquement les transitions des prédateurs
        for agent_id in logprobs_step.keys():
            done = terminated.get(agent_id, False) or truncated.get(agent_id, False)
            buffer.add(
                obs=obs[agent_id],
                action=actions[agent_id],
                logprob=logprobs_step[agent_id],
                reward=rewards[agent_id],
                value=values_step[agent_id],
                done=done
            )

        obs = next_obs

    return buffer, obs



def compute_returns_and_advantages(buffer, gamma=0.99):
    returns = []
    G = 0
    for r, done in zip(reversed(buffer.rewards), reversed(buffer.dones)):
        if done:
            G = 0
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    values = torch.tensor(buffer.values, dtype=torch.float32)
    advantages = returns - values

    return returns, advantages

def ppo_update(policy, optimizer, buffer, returns, advantages, clip_ratio=0.2):
    # Convert everything to tensors
    obs = torch.tensor(buffer.obs, dtype=torch.float32)
    actions = torch.tensor(buffer.actions)
    old_logprobs = torch.tensor(buffer.logprobs)
    advantages = advantages.detach()
    returns = returns.detach()

    # Forward pass
    logits, values = policy(obs)
    dist = torch.distributions.Categorical(logits=logits)
    new_logprobs = dist.log_prob(actions)

    # Probability ratio
    ratio = torch.exp(new_logprobs - old_logprobs)

    # PPO objective
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    # Value loss (MSE)
    value_loss = (returns - values.squeeze()).pow(2).mean()

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()

max_steps = 100
env = simple_tag_v3.parallel_env(
    num_good=1,
    num_adversaries=3,
    num_obstacles=2,
    max_cycles=max_steps,
    continuous_actions=False
)
obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

policy = PPOActorCritic(obs_dim=16)
# policy.to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

for iteration in range(1000):
    buffer, last_obs = collect_trajectories_multi(env, policy, prey_agent, rollout_len=1024)
    returns, advantages = compute_returns_and_advantages(buffer)
    ploss, vloss = ppo_update(policy, optimizer, buffer, returns, advantages)
    buffer.clear()
    
    print(f"Iter {iteration} | policy loss: {ploss:.3f} | value loss: {vloss:.3f}")

torch.save(policy.state_dict(), "predator_model.pth")
