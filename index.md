---
layout: default
title: Survey of Multi-Agent Reinforcement Learning Algorithms
description: "Mini-project #1 for CMSC 818B: Decision-Making for Robotics (F24)"
---

[1] Overview document: https://arxiv.org/pdf/2409.03052 

## Introduction to MARL

Multi-Agent Reinforcement Learning (MARL) is an extension of traditional Reinforcement Learning (RL) that deals with multiple agents learning to interact in a shared environment (Buşoniu et al., 2010).

1. **Primer on normal RL**:
   - RL involves an agent learning to make decisions by interacting with an environment.
   - The agent receives rewards or penalties based on its actions and aims to maximize cumulative rewards (Sutton & Barto, 2018).

2. **Why do we need MARL?**:
   - Real-world scenarios often involve multiple decision-makers or agents.
   - MARL allows for modeling complex interactions and dependencies between agents (Zhang et al., 2021).

3. **What necessitates MARL?**:
   - Scenarios with multiple autonomous entities (e.g., robotics, game theory, traffic control).
   - Problems where decentralized decision-making is crucial (Hernandez-Leal et al., 2019).

4. **How does MARL distinguish itself from traditional RL?**:
   - Multiple agents learning simultaneously.
   - Non-stationary environments due to changing behaviors of other agents.
   - Potential for cooperation, competition, or mixed scenarios (Gronauer & Diepold, 2022).

## Formalization of MARL

MARL can be formalized as an extension of the Markov Decision Process (MDP) used in single-agent RL. MARL typically uses a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) or a Stochastic Game framework (Oliehoek & Amato, 2016; Shoham & Leyton-Brown, 2008).

1. **Dec-POMDP**:
   A Dec-POMDP is defined by a tuple ($I$, $S$, $A_{i \in I}$, $O_{i \in I}$, $P$, $R$, $\gamma$) where:
   - $I$ is the finite set of agents
   - $S$ is the set of environmental states
   - $A_i$ is the set of actions available to agent $i$
   - $O_i$ is the set of observations for agent $i$
   - $P: S \times \prod_{i \in I} A_i \times S \rightarrow [0, 1]$ is the state transition function
   - $R: S \times \prod_{i \in I} A_i \rightarrow \mathbb{R}$ is the reward function
   - $\gamma \in [0, 1]$ is the discount factor
   (Bernstein et al., 2002)

2. **Stochastic Game**:
   A stochastic game is defined by a tuple ($I$, $S$, $A_{i \in I}$, $P$, $R_{i \in I}$) where:
   - $I, S, A_i$ are defined as in Dec-POMDP
   - $P: S \times \prod_{i \in I} A_i \times S \rightarrow [0, 1]$ is the state transition function
   - $R_i: S \times \prod_{i \in I} A_i \rightarrow \mathbb{R}$ is the reward function for agent $i$
   (Shapley, 1953)

3. **Policy and Value Functions**:
   - Policy: $\pi_i: O_i \rightarrow \Delta(A_i)$, where $\Delta(A_i)$ is the probability distribution over actions
   - Joint Policy: $\boldsymbol{\pi} = (\pi_1, ..., \pi_n)$
   - State-Value Function: $V^{\mathbf{\pi}}(s)$ = $\mathbb{E}\_{\mathbf{\pi}}\left[\sum\_{t=0}^{\infty} \gamma^t R(s\_t, \mathbf{a}\_t) \mid s_0 = s\right]$
   - Action-Value Function: $Q^{\mathbf{\pi}}(s, \mathbf{a}) = \mathbb{E}\_{\mathbf{\pi}}\left[\sum\_{t=0}^{\infty} \gamma^t R(s\_t, \mathbf{a}\_t) \mid s_0 = s, \mathbf{a}\_0 = \mathbf{a}\right]$

4. **Learning Objective**:
   The goal in MARL is to find the optimal joint policy $\boldsymbol{\pi}^*$ that maximizes the expected cumulative reward for all agents:

   $$\boldsymbol{\pi}^* = \arg\max_{\boldsymbol{\pi}} \mathbb{E}_{\boldsymbol{\pi}}[\sum_{t=0}^{\infty} \gamma^t R(s_t, \mathbf{a}_t)]$$
   (Zhang et al., 2021)

## Algorithms

MARL algorithms can be broadly categorized into two types: On-line and Off-line.

### On-Line MARL

Agents using On-line MARL algorithms learn a policy by directly interacting with the environment and using its experience to improve its behavior. When multiple-agents are involved, this means agents must also learn how to interact with other agents (either as teammates, opponents, or a combination of the two). Some examples of On-line MARL algorithms are:

1. **Value Decomposition Networks (VDN)**:
   - VDN decomposes the team value function into a sum of individual agent value functions (Sunehag et al., 2018).
   - It assumes that the global Q-function can be additively decomposed into individual agent Q-functions.
   - This approach allows for decentralized execution with centralized training.
   - VDN works by summing the Q-values of individual agent actions and using the total as a Q-value for the entire system:

     $$Q_{tot}(\boldsymbol{\tau}, \mathbf{u}) = \sum_{i=1}^{n} Q_i(\tau_i, u_i)$$

     where $Q_{tot}$ is the total Q-value, $\boldsymbol{\tau}$ is the joint action-observation history (trajectory), $\mathbf{u}$ is the joint action, $n$ is the number of agents, and $Q_i$ is the individual Q-value for agent $i$.

2. **QMIX**:
   - QMIX extends VDN by using a mixing network to combine individual agent Q-values (Rashid et al., 2018).
   - It allows for a more complex relationship between individual and team value functions.
   - QMIX ensures that a global argmax performed on the joint action-value function yields the same result as a set of individual argmax operations performed on each agent's Q-values.
   - The QMIX mixing network combines individual Q-values in a non-linear way:

     $$Q_{tot}(\boldsymbol{\tau}, \mathbf{u}) = f(Q_1(\tau_1, u_1), ..., Q_n(\tau_n, u_n))$$

     where $f$ is a monotonic mixing function implemented as a feed-forward neural network with non-negative weights.

3. **Multi-Agent DDPG (MADDPG)**:
   - MADDPG is an extension of DDPG (Deep Deterministic Policy Gradient) for multi-agent scenarios (Lowe et al., 2017).
   - It uses a centralized training with decentralized execution paradigm.
   - Each agent has its own actor and critic, where the critic has access to all agents' observations and actions during training.
   - MADDPG updates its centralized critic through gradient ascent on the expected return:

     $$\nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\mathbf{x}, \mathbf{a} \sim \mathcal{D}} [\nabla_{\theta_i} \mu_i(a_i | o_i) \nabla_{a_i} Q_i^{\mu}(\mathbf{x}, \mathbf{a})|_{a_i = \mu_i(o_i)}]$$

     where:
     - $\theta_i$ are the parameters of agent $i$'s policy
     - $J(\theta_i)$ is the expected return for agent $i$
     - $\mu_i$ is the deterministic policy of agent $i$
     - $o_i$ is the observation of agent $i$
     - $\mathbf{x}$ is the state of the environment
     - $\mathbf{a}$ is the joint action of all agents
     - $Q_i^{\mu}$ is the centralized action-value function for agent $i$
     - $\mathcal{D}$ is the replay buffer

4. **Multi-Agent PPO (MAPPO)**:
   - MAPPO is an extension of PPO (Proximal Policy Optimization) for multi-agent scenarios (Yu et al., 2021).
   - It combines the sample efficiency of PPO with centralized training and decentralized execution.
   - MAPPO uses a centralized value function and decentralized policies for each agent.
   - The key idea is to optimize the following objective for each agent $i$ using the following loss function:

     $$L^{CLIP}(\theta_i) = \hat{\mathbb{E}}_t[\min(r_t(\theta_i)\hat{A}_t, \text{clip}(r_t(\theta_i), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

     where:
     - $\theta_i$ are the policy parameters for agent $i$
     - $r_t(\theta_i)$ is the probability ratio between the new and old policy
     - $\hat{A}_t$ is the estimated advantage function
     - $\epsilon$ is a hyperparameter that constrains the policy update

   - MAPPO uses a centralized critic to estimate the advantage function, which takes into account the global state and actions of all agents.

### Off-Line MARL

Off-line MARL algorithms learn from a fixed dataset of experiences without direct interaction with the environment. Some examples of Off-line MARL algorithms include:

1. **BCQ for Multi-Agent RL (MA-BCQ)**:
   - MA-BCQ adapts the single-agent Batch Constrained Q-learning (BCQ) to multi-agent settings (Yang et al., 2021).
   - It addresses the extrapolation error in offline RL by constraining the learned policy to be close to the behavior policy in the dataset.

2. **Multi-Agent Constrained Policy Optimization (MACPO)**:
   - MACPO extends Constrained Policy Optimization to multi-agent scenarios for offline learning (Yang et al., 2022).
   - It uses a trust region approach to ensure policy improvement while satisfying constraints.

3. **Offline Multi-Agent Reinforcement Learning with Implicit Constraint (OMAIC)**:
   - OMAIC tackles the challenge of distribution shift in offline MARL (Jiang et al., 2022).
   - It introduces an implicit constraint to penalize out-of-distribution actions and encourages in-distribution actions.

## (Time Permitting) In-Depth Exploration

Explore 1 Algorithm from above in-depth

## (Time Permitting) Current SOTA

Find some papers from recent conferences and talk about what is the current SOTA

## References

Bernstein, D. S., Givan, R., Immerman, N., & Zilberstein, S. (2002). The complexity of decentralized control of Markov decision processes. Mathematics of operations research, 27(4), 819-840.

Buşoniu, L., Babuška, R., & De Schutter, B. (2010). Multi-agent reinforcement learning: An overview. In Innovations in multi-agent systems and applications-1 (pp. 183-221). Springer.

Gronauer, S., & Diepold, K. (2022). Multi-agent deep reinforcement learning: a survey. Artificial Intelligence Review, 55(2), 895-943.

Hernandez-Leal, P., Kartal, B., & Taylor, M. E. (2019). A survey and critique of multiagent deep reinforcement learning. Autonomous Agents and Multi-Agent Systems, 33(6), 750-797.

Jiang, M., Dmitriev, A., & Kuhnle, A. (2022). Offline multi-agent reinforcement learning with implicit constraint. arXiv preprint arXiv:2209.11698.

Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in neural information processing systems (pp. 6379-6390).

Oliehoek, F. A., & Amato, C. (2016). A concise introduction to decentralized POMDPs. Springer.

Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. In International Conference on Machine Learning (pp. 4295-4304). PMLR.

Shapley, L. S. (1953). Stochastic games. Proceedings of the national academy of sciences, 39(10), 1095-1100.

Shoham, Y., & Leyton-Brown, K. (2008). Multiagent systems: Algorithmic, game-theoretic, and logical foundations. Cambridge University Press.

Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., ... & Graepel, T. (2018). Value-decomposition networks for cooperative multi-agent learning based on team reward. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (pp. 2085-2087).

Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.

Yang, Y., Luo, Y., Li, M., Schuurmans, D., & Ammar, H. B. (2022). Batch reinforcement learning with hyperparameter gradients. In International Conference on Machine Learning (pp. 24983-25009). PMLR.

Yang, Y., Tutunov, R., Sakulwongtana, P., & Ammar, H. B. (2021). Project-based multi-agent reinforcement learning. In Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems (pp. 1527-1535).

Yu, C., Velu, A., Vinitsky, E., Wang, Y., Bayen, A., & Wu, Y. (2021). The surprising effectiveness of PPO in cooperative, multi-agent games. arXiv preprint arXiv:2103.01955.

Zhang, K., Yang, Z., & Başar, T. (2021). Multi-agent reinforcement learning: A selective overview of theories and algorithms. Handbook of Reinforcement Learning and Control, 321-384.