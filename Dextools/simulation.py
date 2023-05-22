import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from scipy.stats import halfgennorm, pareto, exponnorm, lognorm, expon
from scipy.special import expit, logit
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def generate_swapped_amount(method = 'exponnorm', past_size = None, iter = None, AR_params = None):
    #Very simple now, to be included: various distributions, some scale operator?
    if method == 'exponnorm':
        while True:
            x = exponnorm.rvs(K=4, loc=150, scale=150, size=1)
            if x > 0:
                return float(x)
            

    if method == "AR":
        AR_params = [0.2274293,  0.13147551, 0.09604017, 
                        0.07126801, 0.06839098, 0.04348023, 
                        0.02150476, 0.01732428, 0.02471407, 
                        0.02150801, 0.00102558, 0.03491916]

        p = len(AR_params)
        if iter <= p:
            while True:
                x = exponnorm.rvs(K=4, loc=150, scale=150, size=1)
                if x > 0:
                    return float(x)
                
        else:
            past_size = np.array(AR_params[-p:])

            while True:
                x = exponnorm.rvs(K=4, loc=75, scale=75, size=1)
                if x >= 0:
                    break

            x = np.sum(past_size * np.array(AR_params)) + float(x)
            return x


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
            #get p past values
            past_swaps = pd.Series(past_swaps[-p:])

            #smoothe out labels
            transformed_series = pd.Series(logit(past_swaps.clip(1e-3, 1 - 1e-3)).values.ravel())
            #print(len(past_swaps), transformed_series)

            #computing the deterministic part based on past values and params
            logit_probablity = np.sum(np.array(transformed_series)*np.flipud(AR_parameters)*0.15)
            #defining random error
            error = np.random.normal(loc=0.0, scale=1.0, size=None)
            #computing probability
            probability = logit_probablity + error
            #print(logit_probablity, expit(logit_probablity), expit(probability), expit(probability) >= 0.5)

            #Return full probability (deterministic + random part). #Here pottential for better effectivity - not expit and compare against 0
            return expit(probability) >= 0.5

def generate_swap_time(method = "expon", herding = False, iter = None, past_times = None):
    if method == "expon":

        if herding == False:
            x = float(expon.rvs(loc = 0, scale=300, size = 1))
            return x
        else:
            scale = 300
            base_scale = 35
            scaling_denominator = scale/(scale-base_scale)
            past_window = 20

            if iter <= past_window:
                x = float(expon.rvs(loc = 0, scale=scale, size = 1))
                return x         
            else:      
                average_scale = np.mean(past_times[-past_window:])/scaling_denominator #possibility to add weights here
                new_time = float(expon.rvs(loc = 0, scale=base_scale, size = 1)) + float(expon.rvs(loc = 0, scale=average_scale, size = 1))
                return new_time

    else:
        raise NameError




def compute_statistics(series):
    data = {
        'Statistic': ['Mean', 'Median', 'St.Dev.', 'Skewness', 'Exc.Kurt.'],
        'Value': [np.mean(series), np.median(series), np.std(series), skew(series), kurtosis(series, fisher=True)]
    }
    df = pd.DataFrame(data)
    return df

def compute_DW_test(returns):
    dw_returns = durbin_watson(returns)
    dw_abs_returns = durbin_watson(np.abs(returns))
    in_range_ret = "*" if 1.5 < dw_returns < 2.5 else ""
    in_range_abs = "*" if dw_abs_returns < 1.5 else ""

    data = {
        'returns': [dw_returns, in_range_ret],
        'absolute returns': [dw_abs_returns, in_range_abs]
    }
    DW_result = pd.DataFrame(data)
    return DW_result

def run_simulation(steps = 1000, x_balance = 1000000, y_balance = 10000):
    # Set the AMM equation
    k = x_balance * y_balance

    # Set up arrays to store results
    times = []
    tokens_swapped = []
    x_balances = []
    y_balances = []
    price = []
    swap_types = []

    # Run simulation for 1000 steps
    for i in range(steps):
        # Generate random time between swaps from exponential distribution
        #time_to_next_swap = float(expon.rvs(loc = 0, scale=300, size = 1))
        time_to_next_swap = generate_swap_time(method="expon", herding = True, iter=i, past_times = times)
        
        # Generate random transaction size from normal distribution
        transaction_size = generate_swapped_amount(method = 'exponnorm', past_size = tokens_swapped, iter=i)
        
        # Generate random transaction type (buy or sell)
        #setting custom AR parameters
        AR_parameter = [0.1,0.08,0.06]
        #AR_parameter = [0.1,0,0,0,0,0,0,0,0,0,0,0.09,0,0,0,0,0,0,0,0,0,0,0.08]
        is_buy = generate_swap_type(method = "AR", past_swaps=swap_types, iter=i, AR_parameters=AR_parameter)

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

    # # Calculate returns
    # arr = np.array(price)
    # returns = np.diff(arr) / arr[:-1]

    return price, times, tokens_swapped, x_balances, y_balances, swap_types

# Transform Tick-by-Tick prices to 5-miute blocks, using seconds in times list
def transform_prices(prices, times):
    start_time = pd.to_datetime('2023-03-19 08:00:00')
    time_deltas = pd.to_timedelta(times, unit='s')
    cumulative_sum = pd.Series(time_deltas).cumsum()
    time_index = start_time + cumulative_sum

    data = {'price': prices}
    df = pd.DataFrame(data, index=time_index)
    df = df.resample('5Min').last().fillna(method='ffill')
    return df

if __name__ == "__main__":
    #Run simulation
    result = run_simulation(steps = 10000)
    price, times, tokens_swapped, x_balances, y_balances, swap_types = result

    # # Calculate TBT returns (not used anymore)
    # arr = np.array(price)
    # returns_tbt = np.diff(arr) / arr[:-1]

    #calculate transformed returns
    price_transformed = transform_prices(price, times)
    returns = price_transformed.price.pct_change().fillna(0)

    print(len(times), len(price) , price_transformed.shape  , returns.shape)

    # Print final balances of tokens in the AMM pool
    print("Final balances:")
    print(f"Token X: {x_balances[-1]}")
    print(f"Token Y: {y_balances[-1]}")
    # Print basic shape statistic of returns
    stats = compute_statistics(returns)
    print(stats)

    # Compute and print DW test Autocorrelation outside of range (1.5, 2.5)
    print("\n")
    DW_result = compute_DW_test(returns)
    print(DW_result)

    # Plot the price and returns
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot X in the first subplot
    axs[0].plot(price)
    axs[0].set_title('Price')

    # Plot Y in the second subplot
    axs[1].plot(returns)
    axs[1].set_title('Returns')
    axs[1].tick_params(axis='x', rotation=45)

    axs[2].hist(returns, bins = 100)
    axs[2].set_title('Returns Histogram')

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    plot_acf(returns, lags=50, ax=ax[0], title = "Returns")
    plot_acf(np.abs(returns), lags=50, ax=ax[1], title = "Absolute Returns")
    limit = 0.6
    ax[0].set_ylim([-1*limit,limit]) 
    ax[1].set_ylim([-1*limit,limit]) 

    plt.tight_layout()
    plt.show()
    

    # print(returns.head(10))
    # print(returns.tail(10))
    