---
layout: default
title: Survey of Multi-Agent Reinforcement Learning Algorithms
description: "Mini-project #1 for CMSC 818B: Decision-Making for Robotics (F24)"
---

[Link to page of what's possible](./possibilities.md).

# Introduction to MARL

Multi-Agent Reinforcement Learning (MARL) is an extension of traditional Reinforcement Learning (RL) that deals with multiple agents learning to interact in a shared environment [1]. Here's an overview of MARL:

1. **Primer on normal RL**: 
   - RL involves an agent learning to make decisions by interacting with an environment.
   - The agent receives rewards or penalties based on its actions and aims to maximize cumulative rewards [2].

2. **Why do we need MARL?**: 
   - Real-world scenarios often involve multiple decision-makers or agents.
   - MARL allows for modeling complex interactions and dependencies between agents [3].

3. **What necessitates MARL?**: 
   - Scenarios with multiple autonomous entities (e.g., robotics, game theory, traffic control).
   - Problems where decentralized decision-making is crucial [4].

4. **How does MARL distinguish itself from traditional RL?**: 
   - Multiple agents learning simultaneously.
   - Non-stationary environments due to changing behaviors of other agents.
   - Potential for cooperation, competition, or mixed scenarios [5].

## Types of MARL
MARL algorithms can be broadly categorized into two types: On-line and Off-line.

### On-Line MARL
On-line MARL algorithms learn from continuous interactions with the environment. Some examples of On-Line MARL algorithms include:

1. **Value Decomposition Networks (VDN)**:
   - VDN decomposes the team value function into a sum of individual agent value functions [6].
   - It assumes that the global Q-function can be additively decomposed into individual agent Q-functions.
   - This approach allows for decentralized execution with centralized training.
   - VDN works by summing the Q-values of individual agent actions and using the total as a Q-value for the entire system:

     $$Q_{tot}(\boldsymbol{\tau}, \mathbf{u}) = \sum_{i=1}^{n} Q_i(\tau_i, u_i)$$

     where $Q_{tot}$ is the total Q-value, $\boldsymbol{\tau}$ is the joint action-observation history (trajectory), $\mathbf{u}$ is the joint action, $n$ is the number of agents, and $Q_i$ is the individual Q-value for agent $i$.

2. **QMIX**:
   - QMIX extends VDN by using a mixing network to combine individual agent Q-values [7].
   - It allows for a more complex relationship between individual and team value functions.
   - QMIX ensures that a global argmax performed on the joint action-value function yields the same result as a set of individual argmax operations performed on each agent's Q-values.
   - The QMIX mixing network combines individual Q-values in a non-linear way:

     $$Q_{tot}(\boldsymbol{\tau}, \mathbf{u}) = f(Q_1(\tau_1, u_1), ..., Q_n(\tau_n, u_n))$$

     where $f$ is a monotonic mixing function implemented as a feed-forward neural network with non-negative weights.

3. **Multi-Agent DDPG (MADDPG)**:
   - MADDPG is an extension of DDPG for multi-agent scenarios [8].
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

### Off-Line MARL
Off-line MARL algorithms learn from a fixed dataset of experiences without direct interaction with the environment. Some examples of Off-line MARL algorithms include:

1. **BCQ for Multi-Agent RL (MA-BCQ)**:
   - MA-BCQ adapts the single-agent Batch Constrained Q-learning (BCQ) to multi-agent settings [9].
   - It addresses the extrapolation error in offline RL by constraining the learned policy to be close to the behavior policy in the dataset.

2. **Multi-Agent Constrained Policy Optimization (MACPO)**:
   - MACPO extends Constrained Policy Optimization to multi-agent scenarios for offline learning [10].
   - It uses a trust region approach to ensure policy improvement while satisfying constraints.

3. **Offline Multi-Agent Reinforcement Learning with Implicit Constraint (OMAIC)**:
   - OMAIC tackles the challenge of distribution shift in offline MARL [11].
   - It introduces an implicit constraint to penalize out-of-distribution actions and encourages in-distribution actions.

# (Time Permitting) In-Depth Exploration
Explore 1 Algorithm from above in-depth

# (Time Permitting) Current SOTA
Find some papers from recent conferences and talk about what is the current SOTA

# References

[1] Buşoniu, L., Babuška, R., & De Schutter, B. (2010). Multi-agent reinforcement learning: An overview. In Innovations in multi-agent systems and applications-1 (pp. 183-221). Springer.

[2] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.

[3] Zhang, K., Yang, Z., & Başar, T. (2021). Multi-agent reinforcement learning: A selective overview of theories and algorithms. Handbook of Reinforcement Learning and Control, 321-384.

[4] Hernandez-Leal, P., Kartal, B., & Taylor, M. E. (2019). A survey and critique of multiagent deep reinforcement learning. Autonomous Agents and Multi-Agent Systems, 33(6), 750-797.

[5] Gronauer, S., & Diepold, K. (2022). Multi-agent deep reinforcement learning: a survey. Artificial Intelligence Review, 55(2), 895-943.

[6] Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., ... & Graepel, T. (2018). Value-decomposition networks for cooperative multi-agent learning based on team reward. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems (pp. 2085-2087).

[7] Rashid, T., Samvelyan, M., Schroeder, C., Farquhar, G., Foerster, J., & Whiteson, S. (2018). QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning. In International Conference on Machine Learning (pp. 4295-4304). PMLR.

[8] Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in neural information processing systems (pp. 6379-6390).

[9] Yang, Y., Tutunov, R., Sakulwongtana, P., & Ammar, H. B. (2021). Project-based multi-agent reinforcement learning. In Proceedings of the 20th International Conference on Autonomous Agents and MultiAgent Systems (pp. 1527-1535).

[10] Yang, Y., Luo, Y., Li, M., Schuurmans, D., & Ammar, H. B. (2022). Batch reinforcement learning with hyperparameter gradients. In International Conference on Machine Learning (pp. 24983-25009). PMLR.

[11] Jiang, M., Dmitriev, A., & Kuhnle, A. (2022). Offline multi-agent reinforcement learning with implicit constraint. arXiv preprint arXiv:2209.11698.