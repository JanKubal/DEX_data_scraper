import numpy as np

# Set initial parameters
x_balance = 10000
y_balance = 10000
k = x_balance * y_balance

# Set up arrays to store results
times = []
tokens_swapped = []
x_balances = []
y_balances = []

# Run simulation for 1000 steps
for i in range(10000):
    # Generate random time between swaps from exponential distribution
    time_to_next_swap = np.random.exponential(scale=1)
    
    # Generate random transaction size from normal distribution
    transaction_size = abs(np.random.normal(loc=30, scale=15))
    
    # Generate random transaction type (buy or sell)
    is_buy = np.random.choice([True, False])
    
    # Calculate new balances based on transaction
    if is_buy:
        x_balance += transaction_size
        y_balance = k / x_balance
    else:
        y_balance += transaction_size
        x_balance = k / y_balance
    
    # Append results to arrays
    times.append(time_to_next_swap)
    tokens_swapped.append(transaction_size)
    x_balances.append(x_balance)
    y_balances.append(y_balance)
    
# Print final balances of tokens in the AMM pool
print("Final balances:")
print(f"Token X: {x_balances[-1]}")
print(f"Token Y: {y_balances[-1]}")