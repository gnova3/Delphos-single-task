## Delphos: A Reinforcement Learning Framework for Discrete Choice Model Specification

Delphos frames the model  specification process as a sequential decision-making problem. Using a Deep Q-Network (DQN) that interacts with an Apollo-based environment, Delphos learns to take a sequence of modelling decision to specify high-performing and behaviourally sound choice models.

For full theoretical details and algorithm design, please refer to our paper: **[Delphos](https://arxiv.org/pdf/2506.06410)**.

<div align="center">
  <img src="img/dqn_framework.png" width="70%" alt="DQN framework for DCM specification">
</div>


In this setting, an agent learns to specify well-performing model candidates by choosing a sequence of modelling actions — such as selecting variables, accommodating both generic and alternative-specific taste parameters, applying non-linear transformations, and including interactions with covariates — and interacting with a modelling environment that estimates each candidate and returns a reward signal. Specifically, Delphos uses a Deep Q-Network that receives delayed rewards based on modelling outcomes (e.g., log-likelihood) and behavioural expectations (e.g., parameter signs), and distributes rewards across the sequence of actions to learn which modelling decisions lead to well-performing candidates.

Delphos thus contains the main cores to define the MDP to learn which modelling decisions lead to better-performing discrete choice models:

- States: Encoded as list of tuple of model components (e.g., variable, transformation, taste type, interaction).
- Actions: Add, modify, or terminate specification components.
- Environment: Translates the agent’s candidate into a model, runs estimation via Apollo, and computes modelling outcomes.
- Rewards: Delayed signals based on estimation quality (e.g., AIC, LL) and behavioural plausibility (e.g., negative cost coefficient).

For questions, suggestions, or collaborations, contact: [Gabriel Nova](mailto:G.N.Nova@tudelft.nl)
