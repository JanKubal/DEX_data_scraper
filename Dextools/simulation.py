import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import halfgennorm, pareto, exponnorm, lognorm
import pandas as pd
from scipy.special import expit, logit

#number of steps
steps = 2000#25000


# Set initial parameters
x_balance = 1000000
y_balance = 10000
k = x_balance * y_balance

# Set up arrays to store results
times = []
tokens_swapped = []
x_balances = []
y_balances = []
price = []
swap_types = []

def generate_swapped_amount(method = 'exponnorm'):
    #Very simple now, to be included: various distributions, some scale operator?
    if method == 'exponnorm':
        while True:
            x = exponnorm.rvs(K=4, loc=100, scale=100, size=1)
            if x > 0:
                return float(x)

def generate_swap_type(method = "random", past_swaps = None, iter = None,
                        AR_parameters = [0.2274293,  0.13147551, 0.09604017, 
                        0.07126801, 0.06839098, 0.04348023, 
                        0.02150476, 0.01732428, 0.02471407, 
                        0.02150801, 0.00102558, 0.03491916]): #taken from MBOX estimation for now
    if method == "random":
        return np.random.choice([True, False])
    
    if method == "AR":
        #taking only last p values
        p = len(AR_parameters)

        if iter <= p:
            return np.random.choice([True, False])
        else:
            past_swaps = pd.Series(past_swaps[-p:])

            transformed_series = pd.Series(logit(past_swaps.clip(1e-3, 1 - 1e-3)).values.ravel())
            #print(len(past_swaps), transformed_series)

            #computing the deterministic part based on past values and params
            logit_probablity = np.sum(np.array(transformed_series)*np.flipud(AR_parameters)*0.15)
            #defining random error
            error = np.random.normal(loc=0.0, scale=1.0, size=None)
            #computing probability
            probability = logit_probablity + error
            #print(logit_probablity, expit(logit_probablity), expit(probability), expit(probability) >= 0.5)

            

            #Here I left of. I added the autoregressive element to the BUY/SELL formation process. #TODO next - analyze what I can infer from the simulation, try some testing (??), write email to LK
            return expit(probability) >= 0.5


# Run simulation for 1000 steps
for i in range(steps):
    # Generate random time between swaps from exponential distribution
    time_to_next_swap = np.random.exponential(scale=1)
    
    # Generate random transaction size from normal distribution
    transaction_size = generate_swapped_amount(method = 'exponnorm')
    
    # Generate random transaction type (buy or sell)
    is_buy = generate_swap_type(method = "AR", past_swaps=swap_types, iter=i)

    # Calculate price of transaction
    price.append(y_balance/x_balance)

    # Calculate new balances based on transaction
    if is_buy:
        x_balance += transaction_size
        y_balance = k / x_balance
    else:
        x_balance -= transaction_size
        y_balance = k / x_balance
    
    # Append results to arrays
    times.append(time_to_next_swap)
    tokens_swapped.append(transaction_size)
    x_balances.append(x_balance)
    y_balances.append(y_balance)
    swap_types.append(int(is_buy))
    
    
# Print final balances of tokens in the AMM pool
print("Final balances:")
print(f"Token X: {x_balances[-1]}")
print(f"Token Y: {y_balances[-1]}")

# Calculate returns
arr = np.array(price)
returns = np.diff(arr) / arr[:-1]

# Plot the price and returns
# Create a figure with two subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot X in the first subplot
axs[0].plot(price)
axs[0].set_title('Price')

# Plot Y in the second subplot
axs[1].plot(returns)
axs[1].set_title('Returns')

axs[2].hist(returns, bins = 100)
axs[2].set_title('Returns Histogram')

# Adjust spacing between subplots
plt.tight_layout()
plt.show()

