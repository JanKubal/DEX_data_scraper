import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import halfgennorm, pareto, exponnorm, lognorm

#number of steps
steps = 25000

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

def generate_swapped_amount():
    #Very simple now, to be included: various distributions, some scale operator?
    while True:
        x = exponnorm.rvs(K=4, loc=100, scale=100, size=1)
        if x > 0:
            return float(x)

# Run simulation for 1000 steps
for i in range(steps):
    # Generate random time between swaps from exponential distribution
    time_to_next_swap = np.random.exponential(scale=1)
    
    # Generate random transaction size from normal distribution
    transaction_size = generate_swapped_amount()
    
    # Generate random transaction type (buy or sell)
    is_buy = np.random.choice([True, False])
    
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

