# btp_pca.py
from dataquery_api import DQInterface
import pandas as pd
import os
from cpa_inputs import cat_dict, cat_dict_short
from cpa_inputs import rates, uncertainty, politics, \
    govt_finances, macro_trends, monetary_policy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def data_visualiser(data_spare,y1,y2,title,xlabel,ylabel,legend1,legend2):
    data_spare[[y1, y2]].plot(figsize=(12, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([legend1, legend2])
    plt.show()

def pca_apply(data, var_list, pcs):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=pcs)
    pca.fit(data_scaled)
    data_pca = pca.transform(data_scaled)
    #show the variance explained by each component
    #print(pca.explained_variance_ratio_)
    #transform data_pca to a dataframe
    data_pca = pd.DataFrame(data_pca)
    #logging.warning('Variance explained: ' + str(pca.explained_variance_ratio_.sum()))
    var_expl = pca.explained_variance_ratio_.sum()
    var_list.append(var_expl)
    #interpret pca components
    # loadings = pca.components_
    # loadings_df = pd.DataFrame(loadings, columns=data.columns)
    # first_pc_loadings = loadings_df.iloc[0]
    # sorted_loadings = first_pc_loadings.abs().sort_values(ascending=False)
    # second_pc_loadings = loadings_df.iloc[1]
    # sorted_loadings = second_pc_loadings.abs().sort_values(ascending=False)
    # third_pc_loadings = loadings_df.iloc[2]
    # sorted_loadings = third_pc_loadings.abs().sort_values(ascending=False)
    # It seems that the first loading is mostly driven by german and italian fiscal data
    # Second = perhaps monetary policy
    return data_pca, var_list

def linreg_apply(data, y):
    model = LinearRegression()
    model.fit(data, y)
    y_pred = model.predict(data)
    return y_pred

def manage_epu(epu: object) -> object:
    # Drop all rows where column 'month' is None so as to remove source
    epu = epu.dropna(subset=['Month'])
    epu['Year'] = epu['Year'].astype(int)
    # get year and month from epu and create a new date column with that year, month, and the LAST day of the month
    epu['real_date'] = pd.to_datetime(epu[['Year', 'Month']].assign(DAY=1))
    epu['real_date'] = epu['real_date'] + pd.offsets.BMonthEnd(0)
    epu_it = epu[['real_date', 'Italy_News_Index']]
    return epu_it

def manage_vix(vix):
    vix['DATE'] = pd.to_datetime(vix['DATE'])
    vix = vix.rename(columns={'DATE': 'real_date', 'CLOSE': 'vix'})
    #drop all columns except real_date and vix
    vix = vix[['real_date', 'vix']]
    return vix

def manage_conf_votes(conf_votes):
    pass

def main():

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 400)

    client_id = "<your_client_id>"  # replace with your client id & secret
    client_secret = "<your_client_secret>"

    client_id = os.getenv("DQ_CLIENT_ID")  # Replace with own client ID
    client_secret = os.getenv("DQ_CLIENT_SECRET")  # Replace with own secret


    # initialize the DQInterface object with your client id & secret
    dq: DQInterface = DQInterface(client_id, client_secret)

    # check that you have a valid token and can access the API
    assert dq.heartbeat(), "DataQuery API Heartbeat failed."

    # # read europe_policy_uncertainty_data.xlsx from same folder and put in dataframe from https://www.policyuncertainty.com/europe_monthly.html
    epu = pd.read_excel('europe_policy_uncertainty_data.xlsx', sheet_name='European News-Based Index')
    # Read VIX from https://www.cboe.com/tradable_products/vix/vix_historical_data/
    vix = pd.read_csv('VIX_History.csv', sep=',')
    conf_votes = pd.read_excel('voti di fiducia.xlsx')

    epu_it = manage_epu(epu)
    vix = manage_vix(vix)
    conf_votes = manage_conf_votes(conf_votes)

    expressions = uncertainty + rates + politics + govt_finances \
                         + macro_trends + monetary_policy

    # dates as strings in the format YYYY-MM-DD
    start_date: str = "2010-01-01"
    end_date: str = "2023-02-05"


    # download the data
    data: pd.DataFrame = dq.download(
        expressions=expressions, start_date=start_date, end_date=None
    )


    #save data to csv



    # We now concatenate all the expressions merging on the real_date column, creating a
    # dataframe which has the real_date column and all the expressions as columns.
    # This is useful for plotting and analysis.

    # make new dataframe where expressions are columns and values are rows, with real_date as index
    data = data.pivot(index="real_date", columns="expression", values="value")

    # If any of the columns have all NaN values, save that column name in a list and drop the column from the dataframe

    # Get list of columns with all NaN values
    all_nan_cols = data.columns[data.isna().all()].tolist()

    # Drop all columns with all NaN values
    data = data.drop(all_nan_cols, axis=1)

    # # Remove all rows where at least one value is NaN
    data = data.dropna(axis=0, how="any")

    data.columns = data.columns.map(lambda x: cat_dict_short.get(x, x))

    # Calculating relevant spreads and r
    data['10y spread'] = data['10y IT'] - data['10y GE']
    data['US Mon Pol'] = data['10y US'] - data['3-month FRA']
    data['GE Mon Pol'] = data['10y GE'] - data['1y GE']
    data = data.reset_index(drop=False)
    data = data.merge(vix, on='real_date', how='left')
    # merge the epu_it dataframe with the data dataframe on the real_date index
    data = data.merge(epu_it, on='real_date', how='left')
    data = data.rename(columns={'Italy_News_Index': 'epu_it'})

    # interpolate all values in data['Italy_News_Index'] column in both directions, so as to also fill the first values
    data['epu_it'] = data['epu_it'].interpolate(method='linear', limit_direction='both', axis=0)
    # data.plot(x='real_date', y=['Italy_News_Index2', 'Italy_News_Index'], style=['-', 'o'])
    # plt.show()
    data = data.set_index('real_date')

    #remove all rows with NaNs
    data = data.dropna(axis=0, how="any")
    # pop out columns we don't need into a new dataframe
    data_spare = data[['10y GE','10y IT', '10y US', '3-month FRA', '10y spread', '1y GE']]
    # drop the columns we don't need from the original dataframe
    data = data.drop(['10y GE','10y IT', '10y US', '3-month FRA', '10y spread', '1y GE'], axis=1)

    correlation_matrix = data.corr()
    # Show correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='seismic', center=0, fmt='.2f', linewidths=.5, cbar=True)
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.show()


    rolling_window = 500
    # Check to see if there are NaNs in the data and tell me which rows have NaNs
    # print(data.isnull().values.any())
    # print(data[data.isnull().any(axis=1)])

    predicted_spreads = []
    # Operate pca_apply on a rolling window of the data
    var_list = []

    for i in range(len(data)-rolling_window+1):
        data_test = data.iloc[0:i+rolling_window].copy()
        data_y_wind = data_spare['10y spread'].iloc[0:i+rolling_window]
        data_pca, var_list = pca_apply(data_test, var_list, pcs=3)
        y_pred = linreg_apply(data_pca, data_y_wind)
        last_date = data_spare.index[i+rolling_window-1]
        data_spare.at[last_date, 'predicted_spread'] = y_pred[-1]

    #data_pca = pca_apply(data, pcs=3)

    # plt.scatter(data_pca[:,0],data_pca[:,1], c = data_spare['10y spread'])
    # plt.show()

    # Use linear regression and build your y_pred
    # model = LinearRegression()
    # model.fit(data_pca, data_spare['10y spread'])
    # y_pred = model.predict(data_pca)
    # data_spare['pred_y'] = y_pred

    # Show the plot of the actual spread vs. the predicted spread
    data_visualiser(data_spare,'10y spread','predicted_spread','BTP-Bund Spread vs. Predicted Spread','Date','Spread Value','BTP-Bund Spread','Predicted Spread')



    # Calculate 30-day rolling st deviation of 10 year spread and predicted spread
    data_spare['10y st dev'] = data_spare['10y spread'].rolling(window=30).std()
    data_spare['10y pred st dev'] = data_spare['predicted_spread'].rolling(window=30).std()
    data_spare['higher chanel'] = data_spare['10y spread'].rolling(window=5).mean() + 1.5*data_spare['10y st dev']
    data_spare['lower chanel'] = data_spare['10y spread'].rolling(window=5).mean() - 1.5*data_spare['10y st dev']


    # Calculate rolling st dev of 10 year spread and predicted spread
    data_spare['10y st dev'] = data_spare['10y spread'].rolling(window=30).std()
    data_spare['10y pred st dev'] = data_spare['predicted_spread'].rolling(window=30).std()

    # Trading strategy

    # if the predicted spread is higher than the upper channel, create a new column
    # called pnl where the pnl will be equal to the spread at time t+1 minus the spread at time t
    # if the predicted spread is lower than the lower channel, create a new column
    # called pnl where the pnl will be equal to the spread at time t minus the spread at time t+1
    # else, create a new column called pnl where the pnl will be equal to 0

    data_spare['PnL bp'] = np.where(data_spare['predicted_spread'] > data_spare['higher chanel'], data_spare['10y spread'].shift(-1, axis = 0) - data_spare['10y spread'], np.where(data_spare['predicted_spread'] < data_spare['lower chanel'], data_spare['10y spread'] - data_spare['10y spread'].shift(-1, axis = 0), 0))

    # plot cumulative pnl
    data_spare['Cumulative PnL'] = data_spare['PnL bp'].cumsum()
    # reset index
    data_spare = data_spare.reset_index(drop=False)
    data_spare.plot(x='real_date', y='Cumulative PnL', kind='line')

    # Show some descriptive statistics for the trading strategy like
    # Sharpe ratio, max drawdown, max drawdown duration, and number of times it correctly predicts the direction of the spread for the next week

    # Sharpe ratio
    # Calculate the daily returns
    data_spare['daily returns'] = data_spare['PnL bp']
    # Calculate the annualised Sharpe ratio
    sharpe_ratio = np.sqrt(252) * data_spare['daily returns'].mean() / data_spare['daily returns'].std()
    print('Sharpe ratio: ', sharpe_ratio)

    data_spare['daily returns'] = data_spare['daily returns'].astype(float)

    # Calculate cumulative returns. This assumes 'daily returns' are already in decimal form (e.g., 0.01 for 1%)
    data_spare['cumulative returns'] = (1 + data_spare['daily returns']).cumprod() - 1
    # Calculate the running maximum of the cumulative returns
    data_spare['running maximum'] = data_spare['cumulative returns'].cummax()
    # Calculate drawdowns as the difference between the running max and the cumulative returns
    data_spare['drawdown'] = data_spare['running maximum'] - data_spare['cumulative returns']
    max_drawdown = data_spare['drawdown'].max()
    print('Maximum Drawdown:', max_drawdown)

    # Max drawdown duration
    # Calculate the duration of the max drawdown
    max_drawdown_duration = data_spare['drawdown'].idxmax()
    print('Max drawdown duration: ', max_drawdown_duration)


    # Sharpe Ratio calculation
    data_spare_copy = data_spare.copy()
    data_spare_copy['real_date'] = pd.to_datetime(data_spare_copy['real_date'])
    data_spare_copy.set_index('real_date', inplace=True)
    data_spare_copy['daily returns'] = data_spare_copy['daily returns'].astype(float)

    sharpe_ratios_by_year = data_spare_copy['daily returns'].groupby(data_spare_copy.index.year).apply(
        calculate_sharpe_ratio)
    sharpe_ratios_by_year = sharpe_ratios_by_year[sharpe_ratios_by_year.index >= 2012]


    # if data_spare['predicted_spread'] > data_spare['higher chanel']:
    #     data['PnL bp'] = data['10y spread'].shift(-1, axis = 0) - data['10y spread']
    # elif data_spare['predicted_spread'] < data_spare['lower chanel']:
    #     data['PnL bp'] = data['10y spread'] - data['10y spread'].shift(-1, axis = 0)
    # else:
    #     data['PnL bp'] = 0



    breakpoint()


    print(data.head())


main()



def calculate_sharpe_ratio(group):
    average_daily_return = group.mean()
    std_dev_daily_return = group.std()
    annualized_return = ((1 + average_daily_return) ** 252) - 1
    annualized_std_dev = std_dev_daily_return * np.sqrt(252)
    R_f = 0  # Risk-free rate
    sharpe_ratio = (annualized_return - R_f) / annualized_std_dev
    return sharpe_ratio




'''
We follow the World Bank's methodology in order to explain asset swap spreads. The determinants are:
EGB Supply
Bank stock outperformance (or credit spreads which we don't have). We will use the government spread of AA-rated 
financials with 3–5 year maturity, as calculated by DB(CREDI,MAGGIE,ECBFAA0305,GS)
'''