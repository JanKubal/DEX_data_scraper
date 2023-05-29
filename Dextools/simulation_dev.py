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
import multiprocessing
from tqdm import tqdm


def generate_swapped_amount():
    #x = lognorm.rvs(s = 1, loc = 0, scale = 200, size = 1) 
    x = lognorm.rvs(s = 0.75, loc = 0, scale = 420, size = 1) 
    #x = lognorm.rvs(s = 1.2, loc = 0, scale = 50, size = 1) #All these variants of lognormal work quite well
    return float(x)


def generate_swap_type(method = "random", past_swaps = None, iter = None, past_times = None,
                        AR_parameters = None, AR_exog_params = None, exog_param_len = None,
                        AR_scale = 0.1, AR_exog_scale = 0.01):
    if method == "random":
        return np.random.choice([True, False])
    
    elif method == "AR" or method == "AR_exog":
        #taking only last p values
        p = len(AR_parameters)

        if iter <= p:
            return np.random.choice([True, False])
        else:
            #get p past values
            past_swaps = np.array(past_swaps[-p:])

            #smoothe out labels
            #transformed_series = logit(past_swaps.clip(1e-3, 1 - 1e-3)).values.ravel()
            transformed_series = logit(np.clip(past_swaps, 1e-3, 1 - 1e-3))#.values.ravel()
            #print(len(past_swaps), transformed_series)

            #computing the deterministic part based on past values and params
            logit_probablity = np.sum(transformed_series*np.flipud(AR_parameters)*AR_scale)
            #defining random error
            error = np.random.normal() #(loc=0.0, scale=1.0, size=None) ##removing explicit statement of default values

            #Adding the exogenous part
            if method == "AR_exog":
                past_times = np.array(past_times[-exog_param_len:])
                past_times_prob_componet = np.sum(past_times*np.flipud(AR_exog_params)*AR_exog_scale)
                
                # if iter % 100 == 0:
                #     print(logit_probablity, past_times_prob_componet, expit(logit_probablity), expit(logit_probablity+past_times_prob_componet))
                logit_probablity += past_times_prob_componet

            #computing probability
            probability = logit_probablity + error
            #print(logit_probablity, expit(logit_probablity), expit(probability), expit(probability) >= 0.5)

            #Return full probability (deterministic + random part).
            #return expit(probability) >= 0.5
            return probability >= 0 ##Here I skip expit conversion to lower performance difficulty


def generate_swap_time(method = "expon", herding = False, iter = None, past_times = None,
                       scale = 150, base_scale = 35, window = 25):
    if method == "expon":

        if herding == False:
            x = float(expon.rvs(loc = 0, scale=300, size = 1))
            return x
        else:
            scale = 200
            base_scale = 50    ###for exponnorm swap ammounts: base_scale = 35, past_window = 20
            scaling_denominator = scale/(scale-base_scale)
            past_window = 50

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
            #scale = 200  #I moved these to function argument
            #base_scale = 50
            scaling_denominator = scale/(scale-base_scale) #the scaling denominator computes the part of the scale that does not come from the base scale
            #window = 50
            
            if iter <= window:
                x = float(pareto.rvs(b = b, loc = -scale, scale = scale, size=1))
                return x
            else:
                average_scale = 2*np.mean(past_times[-window:])/scaling_denominator #the multiplication by 2 is here because in pareto dist, scale = mean*2
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

def run_simulation(steps = 1000, parameters = None):
    # Set the AMM equation
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

    #Defining AR parameters for swap direction
    #AR_parameter = [0.1,0.08,0.06]
    #AR_parameter = [0.1,0,0,0,0,0,0,0,0,0,0,0.09,0,0,0,0,0,0,0,0,0,0,0.08]
    #AR_parameter = [0.2,  0.16, 0.13, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005]
    AR_parameter = [0.1, 0.1, 0.1, 0.1, 0.09, 0.09, 0.09, 0.08, 0.08, 0.07][0:parameters["AR_lags"]]
    exog_param_len = 3#len(AR_parameter)#number of lags considered
    AR_exog_params = [0.0004 - i * (0.0003 / (exog_param_len - 1)) for i in range(exog_param_len)]

    # Run simulation for n steps
    for i in range(steps):
        # Generate random time between swaps from exponential distribution
        #time_to_next_swap = float(expon.rvs(loc = 0, scale=300, size = 1))
        time_to_next_swap = generate_swap_time(method="pareto", herding = True, iter=i, past_times = times, 
                                               scale = parameters["scale"], base_scale = parameters["base_scale"], window = parameters["window"])
        
        # Generate random transaction size from normal distribution
        transaction_size = generate_swapped_amount()
        
        # Generate random transaction type (buy or sell)
        is_buy = generate_swap_type(method = "AR_exog", past_swaps=swap_types, iter=i, AR_parameters=AR_parameter, past_times = times, 
                                    AR_exog_params = AR_exog_params, exog_param_len = exog_param_len,
                                    AR_scale = parameters["AR_scale"], AR_exog_scale = parameters["AR_exog_scale"])

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
        times.append(time_to_next_swap)
        tokens_swapped.append(transaction_size)
        x_balances.append(x_balance)
        y_balances.append(y_balance)
        swap_types.append(int(is_buy))

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

def meta_run(n_times = 10):
    print("Simulation started")
    skewenss_ls, kurtosis_ls = [], []
    dw_returns, dw_abs_returns = [], []
    alphas, alphas_pval = [], []
    betas, betas_pval = [], []
    gammas, gammas_pval = [], []
    for i in range(n_times):
        if i%5 == 0:
            print(f"Run {i}/{n_times}")

        ##Parameters of Scenario 2
        parameters = {"scale": 200, "base_scale": 50, "window": 50, "AR_lags" : 10, "AR_scale" : 0.1, "AR_exog_scale" : -0.05}

        #Trying multiprocessing
        # with multiprocessing.Pool() as pool:
        #     steps = 10000
            
        #     result = pool.starmap(run_simulation, [(steps, parameters) for _ in range(n_times)]) 

        # print(len(result))
        # for i in result:
        #     print(type(i))

        
        
        
        
        result = run_simulation(steps = 10000, parameters = parameters)
        price, times, tokens_swapped, x_balances, y_balances, swap_types = result
        ###To continue here. Create list of parameters and list of results. Save them after each run and create dataframe. Save it in csv.

        #Transforming returns TBT -> 5min
        price_transformed = transform_prices(price, times)
        returns = price_transformed.price.pct_change().fillna(0)
        
        #Skewness and Kurtosis
        skewenss, kurt = skew(returns), kurtosis(returns, fisher=True)
        skewenss_ls.append(skewenss)
        kurtosis_ls.append(kurt)
        #Durbin Watson test for returns and absolute returns
        dw_stat = durbin_watson(returns)
        dw_abs_stat = durbin_watson(np.abs(returns))
        dw_returns.append(dw_stat)
        dw_abs_returns.append(dw_abs_stat)
        #Garch model
        am = arch_model(y = returns, mean='Constant', vol="GARCH", dist='skewt', #'normal', 'studentst', 'skewt',
                        p=1, o=0, q=1, rescale=True)
        res = am.fit(disp="off")
        alphas.append(res.params[2])
        alphas_pval.append(res.pvalues[2])
        betas.append(res.params[3])
        betas_pval.append(res.pvalues[3])
        #HJR GARCH
        am = arch_model(y = returns, mean='Constant', vol="GARCH", dist='skewt', #'normal', 'studentst', 'skewt',
                p=1, o=1, q=1, rescale=True)
        res = am.fit(disp="off")
        gammas.append(res.params[3])
        gammas_pval.append(res.pvalues[3])
    
    print("Simulation finished")
    return skewenss_ls, kurtosis_ls, dw_returns, dw_abs_returns, alphas, alphas_pval, betas, betas_pval, gammas, gammas_pval


def meta_run_parallel(n_times = 10, save_df = False):
    skewenss_ls, kurtosis_ls = [], []
    dw_returns, dw_abs_returns = [], []
    alphas, alphas_pval = [], []
    betas, betas_pval = [], []
    gammas, gammas_pval = [], []
    #Scenario 2
    parameters = {"scale": 200, "base_scale": 50, "window": 50, "AR_lags" : 10, "AR_scale" : 0.1, "AR_exog_scale" : -0.05}
    #Pararell runs
    with multiprocessing.Pool() as pool:
        steps = 10000
        #Use tqdm to create a progress bar
        progress_bar = tqdm(total=n_times)  #Tracking progress - Check for older versions to reverse it back
        def update_progress(*_):
            progress_bar.update(1)
        results = [pool.apply_async(run_simulation, args=(steps, parameters), callback=update_progress)
                            for _ in range(n_times)]
        data_all_runs = [result.get() for result in results]
        progress_bar.close()

    for run in data_all_runs:
        price, times, tokens_swapped, x_balances, y_balances, swap_types = run

        #Transforming returns TBT -> 5min
        price_transformed = transform_prices(price, times)
        returns = price_transformed.price.pct_change().fillna(0)
        
        #Skewness and Kurtosis
        skewenss, kurt = skew(returns), kurtosis(returns, fisher=True)
        skewenss_ls.append(skewenss)
        kurtosis_ls.append(kurt)
        #Durbin Watson test for returns and absolute returns
        dw_stat = durbin_watson(returns)
        dw_abs_stat = durbin_watson(np.abs(returns))
        dw_returns.append(dw_stat)
        dw_abs_returns.append(dw_abs_stat)
        #Garch model
        am = arch_model(y = returns, mean='Constant', vol="GARCH", dist='skewt', #'normal', 'studentst', 'skewt',
                        p=1, o=0, q=1, rescale=True)
        res = am.fit(disp="off")
        alphas.append(res.params[2])
        alphas_pval.append(res.pvalues[2])
        betas.append(res.params[3])
        betas_pval.append(res.pvalues[3])
        #HJR GARCH
        am = arch_model(y = returns, mean='Constant', vol="GARCH", dist='skewt', #'normal', 'studentst', 'skewt',
                p=1, o=1, q=1, rescale=True)
        res = am.fit(disp="off")
        gammas.append(res.params[3])
        gammas_pval.append(res.pvalues[3])   

    if save_df:
        df = df = pd.DataFrame({
                'Scale': [parameters["scale"]] * n_times,
                'Base_scale': [parameters["base_scale"]] * n_times,
                'Window': [parameters["window"]] * n_times,
                'AR_lags': [parameters["AR_lags"]] * n_times,
                'AR_scale': [parameters["AR_scale"]] * n_times,
                'AR_exog_scale': [parameters["AR_exog_scale"]] * n_times,
                'Skewness': skewenss_ls,
                'Kurtosis': kurtosis_ls,
                'DW': dw_returns,
                'DW_absolute': dw_abs_returns,
                'alphas': alphas,
                'alphas_pval': alphas_pval,
                'betas': betas,
                'betas_pval': betas_pval,
                'gammas': gammas,
                'gammas_pval': gammas_pval,
                })

        return df
    else:
        return skewenss_ls, kurtosis_ls, dw_returns, dw_abs_returns, alphas, alphas_pval, betas, betas_pval, gammas, gammas_pval


if __name__ == "__main__":

    #Run simulation and save data as .csv    
    run_n_times = 10
    simulated_data = meta_run_parallel(n_times = run_n_times, save_df=True)
    name = f"Scenario2_{run_n_times}runs"
    file_path = f'D:/Dokumenty/Vejška/Magisterské studium/DIPLOMKA/Code_and_Data/Data_scraping/DEX_data_scraper/simulated_data/{name}.csv'
    #file_path = f"D:/Dokumenty/Vejška/Magisterské studium/DIPLOMKA/Code_and_Data/Data_scraping/DEX_data_scraper/complete_data/{name}/{name}_complete.csv"
    simulated_data.to_csv(file_path)

    testing_metaruns = False
    if testing_metaruns:
        do_parallel_run = True
        if do_parallel_run:
            run_n_times = 10
            skewness_res, kurtosis_res, dw_returns, dw_abs_returns, alphas, alphas_pval, betas, betas_pval, gammas, gammas_pval = meta_run_parallel(n_times = run_n_times)
        
        do_meta_run = False
        if do_meta_run:
            ##Run the metasimulation - Scenario 2
            run_n_times = 10
            skewness_res, kurtosis_res, dw_returns, dw_abs_returns, alphas, alphas_pval, betas, betas_pval, gammas, gammas_pval = meta_run(n_times=run_n_times)

            
        print(f"Skewness mean: {round(np.mean(skewness_res), 5)}, stand.dev: {round(np.std(skewness_res), 5)}")
        print(f"Kurtosis mean: {round(np.mean(kurtosis_res), 5)}, stand.dev: {round(np.std(kurtosis_res), 5)}")
        print("~~~~~~-----------------~~~~~~")
        print(f"DW test mean: {round(np.mean(dw_returns), 5)}, stand.dev: {round(np.std(dw_returns), 5)}")
        print(f"DW absolute mean: {round(np.mean(dw_abs_returns), 5)}, stand.dev: {round(np.std(dw_abs_returns), 5)}")
        print("~~~~~~-----------------~~~~~~")
        print(f"Alphas mean: {round(np.mean(alphas), 5)}, stand.dev: {round(np.std(alphas), 5)}")
        print(f"Betas absolute mean: {round(np.mean(betas), 5)}, stand.dev: {round(np.std(betas), 5)}")
        ##Clearing alphas to only significant
        print("......")
        alphas_signif = [x for x, y in zip(alphas, alphas_pval) if y < 0.05]
        betas_signif = [x for x, y in zip(betas, betas_pval) if y < 0.05]
        print(f"Signif Alphas mean: {round(np.mean(alphas_signif), 5)}, stand.dev: {round(np.std(alphas_signif), 5)}, len {len(alphas_signif)}")
        print(f"Signif Betas absolute mean: {round(np.mean(betas_signif), 5)}, stand.dev: {round(np.std(betas_signif), 5)}, len {len(betas_signif)}")
        print("......")
        gammas_signif = [x for x, y in zip(gammas, gammas_pval) if y < 0.05]
        print(f"Gammas mean: {round(np.mean(gammas), 5)}, stand.dev: {round(np.std(gammas), 5)}")
        print(f"Signif Gammas mean: {round(np.mean(gammas_signif), 5)}, stand.dev: {round(np.std(gammas_signif), 5)}, len {len(gammas_signif)}")




    ##Run a single simulation run for 10000 steps
    single_run = False
    if single_run:
        ##Run simulation
        #Parameters of Scenario 2
        parameters = {"scale": 200, "base_scale": 50, "window": 50, "AR_lags" : 10, "AR_scale" : 0.1, "AR_exog_scale" : -0.05}
        result = run_simulation(steps = 10000, parameters=parameters)
        price, times, tokens_swapped, x_balances, y_balances, swap_types = result

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

        print(f"BUYs: {100*np.sum(swap_types)/10000}%, SELLs: {100*(1-np.sum(swap_types)/10000)}%")


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
    GARCH_switch = False #https://arch.readthedocs.io/en/latest/univariate/introduction.html
    if GARCH_switch:
        am = arch_model(y = returns, mean='Constant', vol="GARCH", dist='skewt', #'normal', 'studentst', 'skewt',
                        p=1, o=0, q=1, rescale=True)
        res = am.fit(disp="off")
        summary_short = pd.concat([res.params, res.pvalues, res.pvalues.apply(lambda x: '*' if x < 0.05 else ' ')], axis=1)

        print("GARCH")
        print(summary_short)
        print(f"Alpha + Beta: {res.params[2]+res.params[3]}")
        print("\n")


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