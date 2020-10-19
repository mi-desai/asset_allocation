def clean_data():

    """
        Returns a list of individual asset time series, all indexed to the same absolute datetime range
        All indices without matching values are NaN's
        Writes all asset time series to csv files in data_clean directory
        Before returning, ensures all indices of all the series are monotonic
        Refactor with list comprehensions
    """

    import numpy as np
    import pandas as pd
    from functools import reduce
    
    data = pd.read_excel('data_raw/asset_data.xlsx')
    
    date_cols = []
    asset_cols = []
    series = []
    absolute_start = '1/3/2005'
    absolute_end = '7/23/2020'
    date_range = pd.date_range(start=absolute_start, end=absolute_end, freq='B')
        
    for column_label in enumerate(data.columns):
        if column_label[0] % 2 == 0:
            date_cols.append(column_label[1])
        else:
            asset_cols.append(column_label[1])
    
    for column in enumerate(date_cols):
        data['Date' + str(column[0] + 1)] = pd.to_datetime(data[column[1]])
        data.drop(column[1], axis=1, inplace=True)
    
    for asset in enumerate(asset_cols):
        columns = data.columns
        modifier = len(asset_cols)
        asset_label = asset[1]
        date_label = columns[asset[0] + modifier]
        new_series = pd.DataFrame(data[[date_label, asset_label]]).set_index(date_label).rename_axis(index='Date').sort_index(ascending=False).dropna()
        new_series = new_series.reindex(date_range).sort_index()
        series.append(new_series)
    
    for asset in enumerate(series):
        asset[1].to_csv('data_standalone/asset' + str(asset[0]+1) + '.csv', index=True)
    
    
    return series


def extract_features(data):
    
    """
        Calculating time-series features for each asset - including returns and risk measurements.
        Went a bit overboard here with all the possible features.
        Lots of inefficient repition in this function, will improve with better data structures, loops and list comprehensions if I have time. 
    """

    import numpy as np
    import pandas as pd
    
    trading_days = 252
    
    for i in enumerate(data):
        
        ts = i[1]
        ts.sort_index()
        colname = ts.columns[0]
        
        # Simple & Rolling Return Measures
        
        # Simple Returns @ Various Lags, as whole numbers, not %
        
        ts['Return_1yr'] = ts[colname].pct_change(freq=str(trading_days) + 'D').mul(100)
        ts['Return_3yr'] = ts[colname].pct_change(freq=str(trading_days * 3) + 'D').mul(100)
        ts['Return_5yr'] = ts[colname].pct_change(freq=str(trading_days * 5) + 'D').mul(100)
        ts['Return_3month'] = ts[colname].pct_change(freq=str(round(trading_days * 0.25)) + 'D').mul(100)
        ts['Return_6month'] = ts[colname].pct_change(freq=str(round(trading_days * 0.50)) + 'D').mul(100)
        ts['Return_9month'] = ts[colname].pct_change(freq=str(round(trading_days * 0.75)) + 'D').mul(100)
        
        # Rolling SMA Returns
        
        ts['SMA_1yr_Return'] = ts['Return_1yr'].rolling(window=str(trading_days) + 'D').mean()
        ts['SMA_3yr_Return'] = ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').mean()
        ts['SMA_5yr_Return'] = ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').mean()
        ts['SMA_3month_Return'] = ts['Return_3month'].rolling(window=str(round(trading_days * 0.25)) + 'D').mean()
        ts['SMA_6month_Return'] = ts['Return_6month'].rolling(window=str(round(trading_days * 0.5)) + 'D').mean()
        ts['SMA_9month_Return'] = ts['Return_9month'].rolling(window=str(round(trading_days * 0.75)) + 'D').mean()
        
        # Cumulative Return
        
        ts['Cum_Return'] = ts[colname].pct_change().add(1).cumprod().sub(1).mul(100)
        
        # Risk Measures
        
        # rolling standard deviations
        ts['STD_1yr_Return'] = ts['Return_1yr'].rolling(window=str(trading_days) + 'D').std()
        ts['STD_3yr_Return'] = ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').std()
        ts['STD_5yr_Return'] = ts['Return_5yr'].rolling(window=str(trading_days * 3) + 'D').std()
        ts['STD_3month_Return'] = ts['Return_3month'].rolling(window=str(round(trading_days * 0.25)) + 'D').std()
        ts['STD_6month_Return'] = ts['Return_6month'].rolling(window=str(round(trading_days * 0.5)) + 'D').std()
        ts['STD_9month_Return'] = ts['Return_9month'].rolling(window=str(round(trading_days * 0.75)) + 'D').std()
        
        # rolling 10-50-90 quantiles
        # rolling interquartile range
        
        ts['Quantile_1yr_10'] = ts['Return_1yr'].rolling(window=str(trading_days) + 'D').quantile(0.1)
        ts['Quantile_1yr_50'] = ts['Return_1yr'].rolling(window=str(trading_days) + 'D').quantile(0.5)
        ts['Quantile_1yr_90'] = ts['Return_1yr'].rolling(window=str(trading_days) + 'D').quantile(0.9)
        ts['IQR_1yr'] = ts['Return_1yr'].rolling(window=str(trading_days) + 'D').quantile(0.75).sub(ts['Return_1yr'].rolling(window=str(trading_days) + 'D').quantile(0.25))
        
        ts['Quantile_3yr_10'] = ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').quantile(0.1)
        ts['Quantile_3yr_50'] = ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').quantile(0.5)
        ts['Quantile_3yr_90'] = ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').quantile(0.9)
        ts['IQR_3yr'] = ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').quantile(0.75).sub(ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').quantile(0.25))
        
        ts['Quantile_5yr_10'] = ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').quantile(0.1)
        ts['Quantile_5yr_50'] = ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').quantile(0.5)
        ts['Quantile_5yr_90'] = ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').quantile(0.9)
        ts['IQR_5yr'] = ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').quantile(0.75).sub(ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').quantile(0.25))
        
        ts['Quantile_3month_10'] = ts['Return_3month'].rolling(window=str(round(trading_days * 0.25)) + 'D').quantile(0.1)
        ts['Quantile_3month_50'] = ts['Return_3month'].rolling(window=str(round(trading_days * 0.25)) + 'D').quantile(0.5)
        ts['Quantile_3month_90'] = ts['Return_3month'].rolling(window=str(round(trading_days * 0.25)) + 'D').quantile(0.9)
        ts['IQR_3month'] = ts['Return_3month'].rolling(window=str(trading_days * 0.25) + 'D').quantile(0.75).sub(ts['Return_3month'].rolling(window=str(trading_days * 0.25) + 'D').quantile(0.25))
        
        ts['Quantile_6month_10'] = ts['Return_6month'].rolling(window=str(round(trading_days * 0.5)) + 'D').quantile(0.1)
        ts['Quantile_6month_50'] = ts['Return_6month'].rolling(window=str(round(trading_days * 0.5)) + 'D').quantile(0.5)
        ts['Quantile_6month_90'] = ts['Return_6month'].rolling(window=str(round(trading_days * 0.5)) + 'D').quantile(0.9)
        ts['IQR_6month'] = ts['Return_6month'].rolling(window=str(trading_days * 0.5) + 'D').quantile(0.75).sub(ts['Return_6month'].rolling(window=str(trading_days * 0.5) + 'D').quantile(0.25))
        
        ts['Quantile_9month_10'] = ts['Return_9month'].rolling(window=str(round(trading_days * 0.75)) + 'D').quantile(0.1)
        ts['Quantile_9month_50'] = ts['Return_9month'].rolling(window=str(round(trading_days * 0.75)) + 'D').quantile(0.5)
        ts['Quantile_9month_90'] = ts['Return_9month'].rolling(window=str(round(trading_days * 0.75)) + 'D').quantile(0.9)
        ts['IQR_9month'] = ts['Return_9month'].rolling(window=str(trading_days * 0.75) + 'D').quantile(0.75).sub(ts['Return_9month'].rolling(window=str(trading_days * 0.75) + 'D').quantile(0.25))

        # rolling skewness at standard lags
        
        ts['Skew_1yr'] = ts['Return_1yr'].rolling(window=str(trading_days) + 'D').skew()
        ts['Skew_3yr'] = ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').skew()
        ts['Skew_5yr'] = ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').skew()
        ts['Skew_3month'] = ts['Return_3month'].rolling(window=str(round(trading_days * 0.25)) + 'D').skew()
        ts['Skew_6month'] = ts['Return_6month'].rolling(window=str(round(trading_days * 0.50)) + 'D').skew()
        ts['Skew_9month'] = ts['Return_9month'].rolling(window=str(round(trading_days * 0.75)) + 'D').skew()
        
        # rolling kurtosis at standard lags
        
        ts['Kurtosis_1yr'] = ts['Return_1yr'].rolling(window=str(trading_days) + 'D').kurt()
        ts['Kurtosis_3yr'] = ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').kurt()
        ts['Kurtosis_5yr'] = ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').kurt()
        ts['Kurtosis_3month'] = ts['Return_3month'].rolling(window=str(round(trading_days * 0.25)) + 'D').kurt()
        ts['Kurtosis_6month'] = ts['Return_6month'].rolling(window=str(round(trading_days * 0.50)) + 'D').kurt()
        ts['Kurtosis_9month'] = ts['Return_9month'].rolling(window=str(round(trading_days * 0.75)) + 'D').kurt()
        
        # rolling min-max range at standard lags
        
        ts['Range_1yr'] = ts['Return_1yr'].rolling(window=str(trading_days) + 'D').max().sub(ts['Return_1yr'].rolling(window=str(trading_days) + 'D').min())
        ts['Range_3yr'] = ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').max().sub(ts['Return_3yr'].rolling(window=str(trading_days * 3) + 'D').min())
        ts['Range_5yr'] = ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').max().sub(ts['Return_5yr'].rolling(window=str(trading_days * 5) + 'D').min())
        ts['Range_3month'] = ts['Return_3month'].rolling(window=str(round(trading_days * 0.25)) + 'D').max().sub(ts['Return_3month'].rolling(window=str(round(trading_days * 0.25)) + 'D').min())
        ts['Range_6month'] = ts['Return_6month'].rolling(window=str(round(trading_days * 0.50)) + 'D').max().sub(ts['Return_6month'].rolling(window=str(round(trading_days * 0.50)) + 'D').min())
        ts['Range_9month'] = ts['Return_9month'].rolling(window=str(round(trading_days * 0.75)) + 'D').max().sub(ts['Return_9month'].rolling(window=str(round(trading_days * 0.75)) + 'D').min())
        
        # Cumulative STD, Quantiles, Skewness, Kurtosis
        
        ts['Cum_STD'] = ts[colname].pct_change().expanding().std()
        ts['Cum_Quantile_10'] = ts[colname].pct_change().expanding().quantile(0.1)
        ts['Cum_Quantile_50'] = ts[colname].pct_change().expanding().quantile(0.5)
        ts['Cum_Quantile_90'] = ts[colname].pct_change().expanding().quantile(0.9)
        ts['Cum_IQR'] = ts[colname].pct_change().expanding().quantile(0.75).sub(ts[colname].pct_change().expanding().quantile(0.25))
        ts['Cum_Range'] = ts[colname].pct_change().expanding().max().sub(ts[colname].pct_change().expanding().min())
        ts['Cum_Skew'] = ts[colname].pct_change().expanding().skew()
        ts['Cum_Kurtosis'] = ts[colname].pct_change().expanding().kurt()
        
        
        ts.index.rename('Date', inplace=True)
        ts.to_csv('data_clean/' + colname + '.csv', index=True)
        
        
def combine_portfolio(data):
    import pandas as pd
    result = pd.DataFrame([])
    for series in data:
        series.index.rename('Date', inplace=True)
        if result.empty == True:
            result = series
        else:
            result = pd.merge(result, series, how='inner', on='Date')     
    return result   


