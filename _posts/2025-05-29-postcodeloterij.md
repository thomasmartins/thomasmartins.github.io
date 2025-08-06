---
title: 'The Psychology Behind the Dutch Postcode Lottery: A Mathematical Exploration'
description: "A behavioral economics and simulation-based analysis of regret, peer pressure, and risk attitudes in the Dutch postcode lottery."
author: "Thomas Martins"
date: 2025-05-29
tags: 
  - Microeconomics
  - Monte Carlo Methods
  - Decision Theory
  - Game Theory
  - Risk and Uncertainty
keywords: ["Microeconomics", "Decision Theory", "Monte Carlo Methods", "Game Theory", "Risk and Uncertainty"]
categories: ["Economics", "Mathematics"]
---

## Introduction

The Postcode Loterij (Dutch Postcode Lottery) is a unique charity lottery in the Netherlands where winning is determined by your postal code. While it has raised significant funds for good causes, it has also drawn criticism for exploiting fear of missing out (FOMO) and social pressure.

In March 2025, Dutch YouTuber Rick Broers, known online as Serpent, launched a citizens’ initiative aiming to ban the Dutch Postcode Lottery in its current form. His campaign, titled “Stop de Postcodeloterij”, targets the lottery’s postcode-based prize system. Broers argues that this system creates social pressure and fear, particularly the fear of seeing one’s neighbors win while not participating oneself.

According to Broers, many people feel forced to participate, not because they enjoy it, but because they are afraid of missing out if their street wins a prize and they don't have a ticket. He emphasizes that this fear (and not desire) is the primary motivation for many participants.

A survey (https://www.hartvannederland.nl/het-beste-van-hart/panel/artikelen/deelnemers-postcodeloterij-spelen-uit-angst-buurt-wint) supports this claim: 42% of players say they participate due to fear. This raises an important question: How much does FOMO affect behavior?

The Postcode Lottery responded to the criticism by defending its format, saying it's unique in that you win together with your neighbors, and that this communal aspect is central to their messaging.


This blog post presents a simple mathematical model to investigate how regret and peer pressure affect the decision to play in the postcode lottery.


## Modeling Decisions: Utility of Income

In order to make a comparison between the decisions of playing and not playing in the lottery, we will need to know how much each agent values gaining or losing money. This will vary with risk preferences. We represent this with an *utility of income* function. For each agent $i$:

$U_i (W,\gamma)$

where $W$ is income and $\gamma$ is some parameter representing risk preferences.

Some of the common forms for the function $U_i$ include

Linear: $U_i(W) = W$. The utility rises linearly (1 per 1) with income. The parameter $\gamma$ is not present, and it represents the agent is indifferent towards risk, or *risk-neutral*. This can represent a real preference but might be deemed irrealistic in many cases.

Logarithmic: $U_i(W) = \ln(W)$. The utility of an additional unit of income ($W$) is always positive, but decreases with the more income you already have. A clear example: if you give `$` 1000 to a poor person, the increase in utility is large but if you give the same `$`1000 to a millionaire the increase in utility will be small (even if still positive). This also does not have a parameter $\gamma$, but represents a preference where the agent does not like risk, or is *risk averse*.

A consequence of this is that the agent will prefer certain outcomes to risky ones with same expected value. Think of the following 2 choices

1. Playing a lottery where you win 100 with probability 50% and 20 with probability 50%
2. Winning 60 with 100% probability

Both have the same expected income ($100 * 50\% + 20 * 50\% = 60$), but if you apply the utility function to the prizes, you will verify the risk-averse agent prefers option 2: $\ln(100) * 50\% + \ln(20) * 50\% = 4.6 * 0.5 + 3 * 0.5 = 3.8 < 4.1 = \ln(60)$

Another type of risk function is the constant relative risk aversion (CRRA) function:

$U_i (W, \gamma) = \frac{W^{1-\gamma}-1}{1-\gamma}$

Where the $\gamma$ controls the degree of risk aversion (for $\gamma=1$ it reduces to the logarithmic function). Larger values of $\gamma$ means the agent is more risk-averse.

```{python}
import numpy as np
import matplotlib.pyplot as plt
```

```{python}
# Define income range
x = np.linspace(1, 10, 500)

# Define CRRA utility function
def crra_utility(x, gamma):
    if gamma == 1:
        return np.log(x)
    else:
        return (x**(1 - gamma) - 1) / (1 - gamma)

# Gamma values to compare
gammas = [0.5, 1.0, 1.5]  # < 1: risk-loving, =1: log, >1: risk-averse

# Plotting
plt.figure(figsize=(10, 6))

for gamma in gammas:
    u = crra_utility(x, gamma)
    label = f"CRRA (γ={gamma})"
    plt.plot(x, u, label=label)

# Reference linear utility
# plt.plot(x, x, label="Risk-Neutral: x", linestyle='--', color="gray")

plt.xlabel("Income")
plt.ylabel("Utility")
plt.title("CRRA Utility Function for Different γ Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

Now that we understand utility functions, we can evaluate how an agent decides between playing or not playing in a lottery. The mechanism is quite simple: they compare the expected utility of both situations, and chooses the highest one. This utility is expected because the outcomes are uncertain (probabilistic). 

$U_{P} > U_{NP}$ means that the agent will play.

### Baseline: The Individual Lottery

For playing in the individual lottery, the expected utility of playing is the utility of the prize amount times the probability of winning minus the utility of the ticket cost. For not playing, it is the utility of zero. Throughout all of this article we shall make the simplifying assumption that the agent's only income is lottery-related.

Play:

$U_P = U(\text{Prize}) * \text{Pr}(\text{Win}) - U(\text{Ticket})$

Don’t play:

$U_{NP} = U(0)$

This constitutes our model for the individual lottery.

### Postcode Lottery Model

Now for the postcode-based lottery, we need to put some additional terms inside the utility functions to model how it works. 

Serpent mentions the abusive marketing practices of the postcode lottery and how much of it goes to charity, but in this article we will not go over that. We shall focus on a mathematical model that compares the postcode lottery to the individual one to assess whether and how fear of missing out (FOMO) and social regret can affect participation.

Serpent says many people play purely out of fear, so we introduce a parameter $\alpha$ to capture that. In turn, the lottery company says there is a positive aspect of playing together with your neighbors, so we also introduce another parameter $\beta$ for that.

$U_{P} = U(\text{Prize}) * \text{Pr}(\text{Win}) - U(\text{Ticket}) + \beta_i$

$U_{NP} = - \alpha _i$

$\alpha$: disutility if neighbors win and agent $i$ didn’t play

$\beta$: utility/disutility of "playing together with your neighbors" i.e. conforming to social norms

Agent will play if $U_i (\text{Play}) > U_i (\text{Don't play})$. Unlike the individual lottery, $\alpha$ and $\beta$ also affect the decision. 

In case agent plays, we need to break it down if it was due to fear of regret or pressure to conform (controlled by $\alpha$ and $\beta$ respectively). This is why we also estimate the individual lottery where $\alpha$ and $\beta$ are both set to zero.

$\alpha$ is calculated as the monetary value corresponding to an utility loss in case the neighbors win but the agent doesn't play. $\beta$ is drawn from the interval $(-10,20)$ in order to represent the utility gain from playing with your neighbors. It can be negative because some people are antisocial and hate their neighbors (many such cases), but for most of the time it's positive.

Agents can be risk-averse, risk-neutral or risk-loving, and the parameter $\gamma$ controls risk preferences. We assume a functional form for utility that is linear for small values (the cost of the ticket), but can reflect risk aversion (for $\gamma < 1$) or risk seeking (for $\gamma >1$). We draw $\gamma$ from the interval $(0.9, 1.1)$.

We calibrate $\alpha$ to reach the 42% of people playing out of fear reported in this news article: https://www.hartvannederland.nl/het-beste-van-hart/panel/artikelen/deelnemers-postcodeloterij-spelen-uit-angst-buurt-wint. We find that the monetary value of the fear of not playing can be as high as 75% of the lottery prize.

We assume regret in case of neighbors winning, peer pressure and risk preferences are completely independent of each other i.e. $\alpha$, $\beta$ and $\gamma$ are independent of each other.

Modeling utility is important because risk preferences are the only factor determining whether agents play or not in the non-postcode lottery, and therefore will constitute our counterfactual for calculating how many people play in the postcode lottery due to fear and/or peer pressure only.

## Simulation Results

We set up our simulation with:

    1000 neighborhoods,

    50 agents each,

    Real-world ticket cost: €31,

    Prize: €30,000,

    Win probability: 0.1%
    
We calibrate the model parameters so they align with the real-world figure from Hart van Nederland: 42% play out of fear.

We also plot how combinations of $\alpha$ and $\beta$ affect play decisions.

```{python}
# Parameters
num_neighborhoods = 1000
agents_per_neighborhood = 50
num_simulations = 1000
lottery_win_prob = 0.001  # low probability of postcode win
lottery_prize = 30000
ticket_cost = 31 # this is the value cited by Rick Broers in his video
# this has to be higher than prize * prob_win because it has to be profitable for the lottery company
```

```{python}
# Social behavior parameters (randomized per agent)
# we pick alpha so it results in 42% playing due to fear
beta_range = (-10, 20)   # peer influence (positive = conformity), assuming there are more people who give in to peer pressure rather than defy it
gamma_range = (0.9, 1.1) # risk preference parameter
alpha_range = (0, 22500) # 75% of prize

def utility(x, gamma):
    c = 50
    if x < c: 
        return x
    else: 
        return c**(1-gamma) * x**gamma
```

```{python}
np.random.seed(123)

postcode_stats = {
    "play_due_to_postcode_system": 0,
    "play_due_to_expected_value": 0,
}
```

```{python}
for _ in range(num_simulations):
    for _ in range(num_neighborhoods):
        alpha = np.random.uniform(*alpha_range, agents_per_neighborhood)
        beta = np.random.uniform(*beta_range, agents_per_neighborhood)
        gamma = np.random.uniform(*gamma_range, agents_per_neighborhood)

        for i in range(agents_per_neighborhood):
            prize_EU = lottery_win_prob * utility(lottery_prize, gamma[i])
            ticket_EU = utility(ticket_cost, gamma[i])
            # regret_cost = alpha[i]
            regret_cost = lottery_win_prob * utility(alpha[i], gamma[i])
            peer_effect = beta[i]

            utility_play = prize_EU - ticket_EU + peer_effect
            utility_dont_play = -regret_cost

            if utility_play >= utility_dont_play:
                if (prize_EU - ticket_EU) < 0: # in this case they would not play without postcode system
                    postcode_stats["play_due_to_postcode_system"] += 1
                else:
                    postcode_stats["play_due_to_expected_value"] += 1
```

```{python}
# Output stats
print("Total simulations:", num_simulations * num_neighborhoods * agents_per_neighborhood)
print("Plays due to postcode system (fear/peer pressure):", postcode_stats["play_due_to_postcode_system"])
print("Plays due to expected value:", postcode_stats["play_due_to_expected_value"])
print("% driven by fear:", (100 * postcode_stats["play_due_to_postcode_system"] / (postcode_stats["play_due_to_postcode_system"] + postcode_stats["play_due_to_expected_value"])))
```

We can also create a data visualization to see what combinations of $\alpha$ and $\beta$ would result in a given percentage of agents playing because of fear/pressure.

```{python}
import seaborn as sns

# Reuse same parameter setup as your model
alpha_vals = np.linspace(0, 20, 30)
beta_vals = np.linspace(-10, 20, 50)
ticket_cost = 31
lottery_prize = 30000
lottery_win_prob = 0.001
c = 50

# Utility function as in your code
def utility(x, gamma):
    if x < c: 
        return x
    else: 
        return c**(1-gamma) * x**gamma

# Grid to store proportions of fear-based play
fear_matrix = np.zeros((len(alpha_vals), len(beta_vals)))

# Simulation
agents_per_combo = 10000

for i, alpha in enumerate(alpha_vals):
    for j, beta in enumerate(beta_vals):
        fear_count = 0
        for _ in range(agents_per_combo):
            gamma = np.random.uniform(0.9, 1.1)

            prize_EU = lottery_win_prob * utility(lottery_prize, gamma)
            ticket_EU = utility(ticket_cost, gamma)
            regret_cost = alpha
            peer_effect = beta

            utility_play = prize_EU - ticket_EU + peer_effect
            utility_dont_play = -regret_cost

            if utility_play >= utility_dont_play and (prize_EU - ticket_EU) < 0:
                fear_count += 1

        fear_matrix[i, j] = fear_count / agents_per_combo

# Create heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(fear_matrix, xticklabels=np.round(beta_vals, 1), yticklabels=np.round(alpha_vals, 1), cmap="viridis")
plt.xlabel("Peer Influence (β)")
plt.ylabel("Regret Sensitivity (α)")
plt.title("Proportion Playing Due to Fear\nby α (regret) and β (peer influence)")
plt.tight_layout()
plt.show()
```

Conclusion

Our model supports Rick Broers' claim: a significant portion of players are motivated not by hope, but by fear and social pressure.

Key insights:

    The postcode lottery creates a form of social regret that distorts rational decision-making.

    A large regret cost (up to 75% of the prize) is required to explain observed behavior.

    The model could be expanded with spatial dynamics, network effects, or evolving beliefs.

We hope this exploration sparks further ethical and academic discussions on how lotteries influence human behavior.

