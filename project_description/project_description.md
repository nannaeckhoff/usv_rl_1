# Safety and Risk Representation in Reinforcement Learning for Autonomous USV Navigation

## Project Description

### First – Some Terminology

In this project, **safety** refers to undesirable quantitative behaviour, such as a physical collision or entering an unsafe region. A safety-aware formulation is designed to avoid or reduce such behaviour, for example by maintaining distance from obstacles or penalizing collision-related behaviour.

**Risk** refers to the likelihood and severity of undesirable outcomes when the underlying data are uncertain. A risk-aware formulation therefore accounts for uncertainty when making decisions, for example by considering the potential for catastrophic outcomes under uncertain observations or dynamics.

The initial project focuses exclusively on safety. The environment and observations are assumed to be deterministic and perfectly known. Uncertainty and risk-aware formulations are considered as possible extensions.

### Project Description

This project investigates how safety and risk can be represented and incorporated into reinforcement learning (RL) for autonomous navigation of an unmanned surface vessel (USV).

The objective is to use a simple and controlled simulation environment to systematically study how different safety formulations affect the behaviour and performance of an RL agent. The focus is therefore on the RL formulation rather than realistic USV dynamics.

The project follows the progression:

```
0. Literature review
        ↓
1. Baseline navigation environment
        ↓
2. Establish Evaluation Metrics
        ↓
3. Compare safety representations
        ↓
4. Investigate safety weighting
        ↓
5. Uncertainty & risk (Extension)
```

## Initial Scope

The initial navigation problem is deliberately simple. An RL agent must navigate towards a goal, with obstacles introduced incrementally. A simple kinematic USV representation is sufficient. Realistic vehicle dynamics are outside the scope of the initial project because the safety objective (obstacle avoidance) is related to the vessel kinematics (velocity and spatial position) rather than its dynamics. Integrating dynamics can make the results less generalizable as they become dependent on the dynamics modeling. However, this may be considered as an extension for later on (e.g., if we want to do reward shaping) in which case we could work directly with Rabea's simulator environment.

The main research question is: *How does the representation and formulation of safety affect the behaviour and performance of reinforcement-learning-based USV navigation?*

The project addresses two related questions:

1. How does the representation of safety affect the learned navigation policy?
2. How does the relative weighting of navigation performance and safety affect the resulting trade-off, and are some safety representations more sensitive to this weighting?

## Task 0: Literature Review

The first part of the project is to conduct a literature review to understand how safety has been represented and studied in RL navigation, and use this to identify relevant approaches for the systematic experiments. To this end, find and read literature on

1. RL for USV/ASV navigation and general autonomous navigation.
2. Safety-aware and constrained RL.
3. Representations of safety in the RL reward/constraints, such as distance-based, collision-based (binary), and velocity/direction-dependent formulations.

The review should identify how safety can be represented, how it can be incorporated into RL, and how safety and navigation performance can be evaluated systematically. Summarize your review in a LaTeX document to be used in the following steps.

Note that there is a large body of work on reward shaping, where rewards are designed or modified to encourage specific behaviours. The aim of this study is to conduct a systematic comparison of different ways of representing safety in a more general setting, while keeping the rest of the RL formulation as consistent as possible. Therefore, focus on identifying generally applicable safety formulations rather than application-specific reward shaping.

## Task 1: Establish a Baseline Navigation Environment

Implement a simple navigation environment using Safety-Gymnasium (https://github.com/openai/safety-gym). The environment should initially contain only a USV and a goal, with simple kinematic motion and no obstacle.

Configure a standard RL algorithm (Hanna recommended stable-baselines, cleanRL, and spinningUp). Establish a baseline navigation policy. The purpose of this task is to verify that the environment and training procedure work reliably before introducing safety considerations.

A static obstacle can then be introduced incrementally to provide the basic setting for the safety experiments.

## Task 2: Establish Benchmarking and Evaluation Metrics

The next part of the project is to establish the benchmarking and evaluation metrics because we need a consistent way to measure and compare both navigation performance and safety across the different experiments.

Each experimental condition should be evaluated using multiple independent random seeds to account for variability in RL training (3-5 seeds are likely sufficient for an initial comparison).

Evaluation should consider both navigation and safety metrics, such as:

- goal-reaching success rate,
- navigation performance,
- collision or constraint violation rate,
- cumulative safety cost.

Results should be reported across random seeds using appropriate summary statistics, such as mean and standard deviation. The same evaluation scenarios should be used when comparing different safety representations and weighting choices.

The benchmarking procedure should be defined consistently across experiments so that differences in performance can be attributed to the safety formulation rather than differences in training or evaluation conditions.

## Task 3: Compare Safety Representations

Implement a number of systematically selected safety representations while keeping the RL algorithm, environment, training procedure, and other experimental conditions fixed.

The representations should be selected based on the literature review and include both standard (simple) reward formulations and less standard ones (e.g., velocity-oriented). It is especially interesting to examine the performance under less standard representations to find out whether they prove advantageous over the more simple formulations in the current setting.

The objective is to determine how different representations influence the learned policy and the resulting navigation-safety trade-off based on the evaluation metrics defined earlier.

Summarize and discuss the results in a systematic and reproducible manner.

## Task 4: Investigate Safety Weighting

For selected safety representations, vary the relative importance assigned to safety in the learning objective.

Evaluate how changes in this weighting affect navigation performance and safety. This allows the sensitivity of different safety representations to be compared systematically. (See details in the earlier draft.)

## Task 5: Uncertainty and Risk (Extension)

If time permits, introduce uncertainty into the observations or environment and investigate whether risk-aware formulations provide advantages over the safety formulations studied in the initial project.
