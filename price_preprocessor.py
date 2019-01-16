import glob
#import tensorflow as tf
import pandas as pd
import numpy as np
#from matplotlib import pyplot as plt
#from sklearn import metrics

import os
import csv
from timeit import default_timer
from datetime import date
# dev tools
'''
import cProfile
import pstats
'''

def filtered_df(df_raw, day_as_profit, for_prediction, min_value_traded, max_daily_change, max_profit_change):
	# check if value traded is enough
	df_raw["is_val_traded_enough"] = (df_raw["Volume"] * df_raw["Close"]) > min_value_traded

	# check if max_daily_change threshold is not passed
	arr_temp = df_raw["Close"][:-1].values
	arr_temp = np.insert(arr_temp, 0, 0)
	df_raw["prev_close"] = arr_temp
	df_raw["is_daily_change_below_max"] = abs((df_raw["Close"] / df_raw["prev_close"]) - 1) < max_daily_change
	
	# check if profit (& open?) does not pass max_daily_change
	arr_temp = df_raw["Close"][day_as_profit:].values
	arr_temp = np.append(arr_temp, np.tile([None], day_as_profit))
	#print(arr_temp.shape, df_raw.shape)
	df_raw["tp_day_close"] = arr_temp
	df_raw["is_profit_change_normal"] = (abs((arr_temp / df_raw["Close"]) - 1) < max_profit_change) | (arr_temp == None)
	#print(df_raw)
		
	# check if price change happens
	df_raw["is_price_not_flat"] = df_raw["Low"] != df_raw["High"]

	# consolidate
	df_raw["is_ok"] = (
		df_raw["is_val_traded_enough"] &
		df_raw["is_daily_change_below_max"] &
		df_raw["is_profit_change_normal"] &
		df_raw["is_price_not_flat"]
		)

	df_raw["is_today_ok"] = df_raw["is_ok"]
	df_raw["next_day_open"] = df_raw["Open"].shift(-1)
	df_raw["tp_day_d+%i" % day_as_profit] = df_raw["Close"].shift(-day_as_profit)

	if not for_prediction:
		df_raw.dropna(axis=0, inplace=True)

	# cleanup
	df_raw.drop(columns=[
			'Adj Close',
			'is_val_traded_enough', 
			'prev_close', 
			'is_daily_change_below_max', 
			'tp_day_close', 
			'is_profit_change_normal', 
			'is_price_not_flat'
			], inplace=True)
	#print(df_raw.columns)
	return df_raw

def pct_processor(shuffle=True, for_prediction=False):
	'''
	Process the flattened data to percentage, compiled to one file
	'''
	# TODO: add stock name to data
	print("Start processing...")
	start_time = default_timer()

	path = r'./preprocessed_data/'
	if for_prediction:
		path = r'./prediction_data/'

	all_files = glob.glob(path + "/*.csv")
	#all_files = ['./preprocessed_data/AALI.csv', './preprocessed_data/PGAS.csv']

	columns = list(pd.read_csv(all_files[0]).columns.values)
	columns.insert(0, "ticker_name")

	#if not for_prediction:
	#	tp_day_col_name = columns[len(columns)-2]
	#	day_as_profit = int(tp_day_col_name.split("+")[1])
	
	df = pd.DataFrame(columns=columns)
	
	for file_ in all_files:
		ticker_name = os.path.basename(file_)[:4]
		df_raw = pd.read_csv(file_)
		df_temp = df_raw.drop(["Date", "Volume"], axis=1)

		divisor = df_temp["O0"]
		df_temp = df_temp.apply(lambda x: x / divisor)
		for col in df_temp.columns:
			df_temp[col] = df_temp[col].apply(lambda x: (-1/x)+1 if x < 1 else x - 1)

		df_raw["ticker_name"] = ticker_name

		df_temp = pd.concat([
			df_raw[["ticker_name", "Date"]],
			pd.DataFrame(df_temp),
			df_raw["Volume"]
			], axis=1)
		
		df = df.append(df_temp, ignore_index=True, sort=False)
	
	if shuffle:
		df = df.reindex(np.random.permutation(df.index))

	print(df.describe())
	print(df.head())
	if for_prediction:
		prediction_path = "./prediction_data/compiled_per_date/"
		today_str = pd.read_csv(all_files[0])["Date"][0]
		df.to_csv("%sprediction_data_%s.csv" % (prediction_path, today_str), index=False)
	else:
		df.to_csv("preprocessed_data.csv", index=False)

	duration = default_timer() - start_time
	print(f"Finished processing {len(all_files)} files in {duration:.02f} s")

def OHLC_flattener(n_days, day_as_profit, for_prediction=False, min_value_traded=750000000, max_daily_change = 0.26):
	"""
	Outputs pandas dataframe with OHLC of x days and profit column
	Args:
		n_days			: number of days as features
		day_as_profit	: day from which profit data is taken after n_days
		min_value_traded: min market value from each feature day (default: 750 mio)
	"""
	print("Start processing...")
	start_time = default_timer()

	path = r'./IDX'
	save_path = "./preprocessed_data/"
	if for_prediction:
		save_path = "./prediction_data/"
		
	all_files = glob.glob(path + "/*.csv")
	#all_files = ['./IDX/AALI.JK.csv', './IDX/PGAS.JK.csv']
	#print(all_files)

	
	# iterate stocks
	max_profit_change = ((max_daily_change + 1) ** day_as_profit) - 1
	for file_ in all_files:
		
		if for_prediction:
			with open(file_, "r") as f:
				r = csv.reader(f)
				row_count = sum(1 for row in r) - 1
				#print(range(1, row_count-n_days))
			df_temp = pd.read_csv(file_, skiprows=range(1, row_count-n_days))
			#print(df_temp)
		else:
			df_temp = pd.read_csv(file_)


		#print(df_raw)
		ticker_name = os.path.basename(file_)[:4]
		#print("Processing", ticker_name + "...")

		#df = pd.DataFrame(columns=columns)
		#df.set_index(["ticker", "date_start"])
		#print(df)
		

		# make bool columns for future evaluation
		df_temp = filtered_df(df_temp, day_as_profit, for_prediction, min_value_traded, max_daily_change, max_profit_change)

		# make "is_ok" with history, historical candlestick, and column names
		flag_temp = df_temp["is_ok"].values

		arr_temp = np.array([]).reshape(len(df_temp.index), 0)
		arr_append = df_temp[["Open", "High", "Low", "Close"]].values
		none_arr = [0, 0, 0, 0]

		columns = []
		for i in range(n_days):
			# price
			arr_temp = np.concatenate((arr_append, arr_temp), axis=1)
			#print(arr_temp)
			#print(arr_append)
			arr_append = np.insert(arr_append, 0, none_arr, axis=0)
			arr_append = arr_append[:-1]

			# "is_ok" flag
			df_temp["is_ok"] = df_temp["is_ok"] & flag_temp
			flag_temp = np.insert(flag_temp, 0, False)
			flag_temp = flag_temp[:-1]

			# create column
			for ch in "OHLC":
				columns.append(ch+str(i))
		
		df_temp = pd.concat([
			df_temp["Date"],
			pd.DataFrame(arr_temp, columns=columns),
			df_temp[["next_day_open",df_temp.columns[-1], "Volume", "is_ok", "is_today_ok"]]
			], axis=1)
		
		#print(df_temp)
		df_temp.drop(df_temp[df_temp["is_ok"] == False].index, inplace=True) # drop all not okay data
		df_temp.drop(columns=["is_ok", "is_today_ok"], inplace=True) # drop boolean filter
		
		# check if enough ok transaction per ticker
		row_count = df_temp.shape[0]
		if for_prediction:
			df_temp.to_csv(save_path + ticker_name + ".csv", index=False)
		else:
			if row_count > 120:
				#print(df)
				df_temp.to_csv(save_path + ticker_name + ".csv", index=False)
			#else:
			#	print("Skipped", ticker_name)

	duration = default_timer() - start_time
	print(f"Finished processing {len(all_files)} files in {duration:.02f} s")
	# filter based on value traded

def prediction_normalize():
	price_path = "./prediction_data/"
	path = glob.glob("./prediction_data/prediction_history/*")
	file_path = max(path, key=os.path.getctime)

	df = pd.read_csv(file_path)
	df = df[df.columns[:-2]]
	open0_col = "O0"
	df_open0 = pd.DataFrame(np.nan, index=range(df.shape[0]),columns=[open0_col])

	# check if date is same
	ticker_name = df.loc[0, "ticker_name"]
	date_source = pd.read_csv(price_path + ticker_name + ".csv").loc[0, "Date"]
	if date_source != df.loc[0, "Date"]:
		raise Exception("Date source and prediction data not equal!\n%s | %s" % (date_source, df.loc[0, "Date"]))

	for i, row in df.iterrows():
		ticker_name = row["ticker_name"]
		open0 = pd.read_csv(price_path + ticker_name + ".csv").loc[0, open0_col]
		df_open0.at[i, open0_col] = open0

	cols_dont_multiply = ["ticker_name", "Date", "prediction_clusters", "exp_profit_pct"]
	error_cols = ["RMSE_next_open_day", "RMSE_tp_day", "diff_with_cluster_centers"]
	for col in df.columns:
		if col not in cols_dont_multiply:
			if col in error_cols:
				df[col] = df[col] * df_open0[open0_col]
			else:
				df[col] = (df[col] + 1) * df_open0[open0_col]
	#print(df)
	#print(df[["O0", "predict_next_open_day", "predict_tp_day"]])
	#print(df.columns)

	df.to_csv("./figures/latest_prediction.csv", index=False)

#OHLC_flattener(3, 2, for_prediction=True)
#pct_processor(for_prediction=True)
prediction_normalize()

# debug
'''
debug_file_name = 'cProfile'
cProfile.run('OHLC_flattener(3, 1)', debug_file_name)
p = pstats.Stats(debug_file_name)
p.sort_stats('cumulative').print_stats(25)
'''