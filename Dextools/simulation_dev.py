import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from scipy.stats import halfgennorm, pareto, exponnorm, lognorm, expon
from scipy.special import expit, logit
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model

import time
# Start the timer
start_time = time.time()


def generate_swapped_amount(method = 'exponnorm', past_size = None, iter = None, AR_params = None):
    #Very simple now, to be included: various distributions, some scale operator?
    if method == 'exponnorm':
        while True:
            x = exponnorm.rvs(K=4, loc=150, scale=150, size=1)
            if x > 0:
                return float(x)

    elif method == 'lognorm':
        while True:
            #x = lognorm.rvs(s = 1, loc = 0, scale = 200, size = 1) 
            x = lognorm.rvs(s = 0.75, loc = 0, scale = 420, size = 1) ##TODO before real run - remove the negative conditioning
            #x = lognorm.rvs(s = 1.2, loc = 0, scale = 50, size = 1) #All these variants of lognormal work quite well
            if x > 0:
                return float(x)
    
    elif method == "AR":  ###Dead end, DEPRECATED
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
            base_scale = 35    ###for exponnorm swap ammounts: base_scale = 35, past_window = 20
            scaling_denominator = scale/(scale-base_scale)
            past_window = 25

            if iter <= past_window:
                x = float(expon.rvs(loc = 0, scale=scale, size = 1))
                return x         
            else:      
                average_scale = np.mean(past_times[-past_window:])/scaling_denominator #possibility to add weights here
                new_time = float(expon.rvs(loc = 0, scale=base_scale, size = 1)) + float(expon.rvs(loc = 0, scale=average_scale, size = 1))
                return new_time
            
    elif method == "pareto":
        if herding == False:
            #r = pareto.rvs(b = 3, loc = -150, scale = 150, size=1) 
            #r = pareto.rvs(b = 4, loc = -250, scale = 250, size=1)
            x = float(pareto.rvs(b = 3, loc = -150, scale = 150, size=1))
            return x
        else:
            b = 3
            scale = 150
            base_scale = 35
            scaling_denominator = scale/(scale-base_scale) #the scaling denominator computes the part of the scale that does not come from the base scale
            past_window = 25
            
            if iter <= past_window:
                x = float(pareto.rvs(b = b, loc = -scale, scale = scale, size=1))
                return x
            else:
                average_scale = 2*np.mean(past_times[-past_window:])/scaling_denominator #the multiplication by 2 is here because in pareto dist, scale = mean*2
                new_time = float(pareto.rvs(b = b, loc = -base_scale, scale = base_scale, size=1)) + float(pareto.rvs(b = b, loc = -average_scale, scale = average_scale, size=1))
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
    slippage = []

    # Run simulation for n steps
    for i in range(steps):
        # Generate random time between swaps from exponential distribution
        #time_to_next_swap = float(expon.rvs(loc = 0, scale=300, size = 1))
        time_to_next_swap = generate_swap_time(method="pareto", herding = True, iter=i, past_times = times)
        
        # Generate random transaction size from normal distribution
        transaction_size = generate_swapped_amount(method = 'lognorm', past_size = tokens_swapped, iter=i)
        
        # Generate random transaction type (buy or sell)
        #setting custom AR parameters
        AR_parameter = [0.1,0.08,0.06]
        #AR_parameter = [0.1,0,0,0,0,0,0,0,0,0,0,0.09,0,0,0,0,0,0,0,0,0,0,0.08]
        is_buy = generate_swap_type(method = "AR", past_swaps=swap_types, iter=i, AR_parameters=AR_parameter)

        # Calculate price of transaction
        old_price = y_balance/x_balance
        #price.append(old_price) #I am not sure what to do about slippage

        # Calculate new balances based on transaction
        if is_buy:
            x_balance += transaction_size
            y_balance = k / x_balance
        else:
            x_balance -= transaction_size
            y_balance = k / x_balance

        new_price = y_balance/x_balance
        
        # Append results to arrays
        price.append(new_price)
        slippage.append(new_price/old_price - 1)
        times.append(time_to_next_swap)
        tokens_swapped.append(transaction_size)
        x_balances.append(x_balance)
        y_balances.append(y_balance)
        swap_types.append(int(is_buy))

    return price, times, tokens_swapped, x_balances, y_balances, swap_types, slippage

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
    price, times, tokens_swapped, x_balances, y_balances, swap_types, slippage = result

    # Calculate TBT returns (not used anymore)
    arr = np.array(price)
    returns_tbt = np.diff(arr) / arr[:-1]

    #calculate transformed returns
    price_transformed = transform_prices(price, times)
    returns = price_transformed.price.pct_change().fillna(0)

    #print(len(times), len(price) , price_transformed.shape  , returns.shape)

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


    ## Visual-check plots
    do_plots = False
    if do_plots:
        # Plot the price and returns
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        # Plot X in the first subplot
        axs[0][0].plot(price)
        axs[0][0].set_title('Price')
        # Plot Y in the second subplot
        axs[0][1].plot(returns)
        axs[0][1].set_title('Returns')
        axs[0][1].tick_params(axis='x', rotation=45)
        #Plot histogram of returns
        axs[0][2].hist(returns, bins = 100)
        axs[0][2].set_title('Returns Histogram')
        #Plot ACF and PACF
        plot_acf(returns, lags=50, ax=axs[1][0], title = "Returns", marker = ".")
        plot_acf(np.abs(returns), lags=50, ax=axs[1][1], title = "Absolute Returns", marker = ".")
        limit = 0.5
        axs[1][0].set_ylim([-1*limit,limit]) 
        axs[1][1].set_ylim([-1*limit,limit]) 
        #Plot the center of the histogram
        filtered_returns = returns[(returns >= np.percentile(returns, 1)) & (returns <= np.percentile(returns, 99))]
        axs[1][2].hist(filtered_returns, bins = 100)
        axs[1][2].set_title('Center Histogram')
        plt.tight_layout()
        plt.show()

    ##Estimate GARCH 
    GARCH_switch = True #https://arch.readthedocs.io/en/latest/univariate/introduction.html
    if GARCH_switch:
        am = arch_model(y = returns, mean='Constant', vol="GARCH", dist='skewt', #'normal', 'studentst', 'skewt',
                        p=1, o=0, q=1, rescale=True)
        res = am.fit(disp="off")
        summary_short = pd.concat([res.params, res.pvalues, res.pvalues.apply(lambda x: '*' if x < 0.05 else ' ')], axis=1)

        print("GARCH")
        print(summary_short)
        print(f"Alpha + Beta: {res.params[2]+res.params[3]}")
        print("\n")

    ###display slippage ###---it is return in fact!!
    #print(slippage[0:50])
    # stats_slip = compute_statistics(slippage)
    # print(stats_slip)    
    # plt.hist(slippage, bins=100)
    # plt.show() #Does this have any sense doing? Whats the point?
    # #A negative slippage that traders suffers due to price impact of his swap is positive returns for holders of the bought token



    ###BUY/SELL examination
            # swap_filter = [bool(element) for element in swap_types]

            # sell_prices = np.array(price)[swap_filter]
            # buy_prices = np.array(price)[np.logical_not(swap_filter)]

            # print(len(price), len(buy_prices), len(sell_prices))

            # # print(compute_statistics(sell_prices))
            
            # # print("\n")
            # # print(compute_statistics(buy_prices))


            # start_time = pd.to_datetime('2023-03-19 08:00:00')
            # time_deltas = pd.to_timedelta(times, unit='s')
            # cumulative_sum = pd.Series(time_deltas).cumsum()
            # time_index = start_time + cumulative_sum


            # time_index_sell = time_index[swap_filter]
            # time_index_buy = time_index[np.logical_not(swap_filter)]

            # plt.plot(time_index_sell, sell_prices, label='Sell Prices')
            # plt.plot(time_index_buy, buy_prices, label='Buy Prices')

            # plt.xlabel('Time')
            # plt.ylabel('Price')
            # plt.legend()

            # plt.show()


# End the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the runtime
print(f"Runtime: {elapsed_time} seconds")