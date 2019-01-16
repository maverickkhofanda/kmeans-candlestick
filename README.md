# kmeans-candlestick
An attempt to classify candlestick prices by using TensorFlow k-means, and predict the return and risk in &lt;1 week

Result samples can be seen in the 'figures' folder.

|Files|Description|
|---|---|
|idx_dl_csv.py|Data downloader from Yahoo Finance. Need txt file containing the ticker names in bash command. (Sample: idx_stocks_list.txt)|
|price_preprocessor.py|Filter and preprocess data downloaded from Yahoo Finance. Also contain post processor which isn't moved out yet. (prediction_normalize() function)|
|kmeans.ipynb|For training cluster centers. Outputs cluster centers, RMSE and expected profit in csv files.|
|kmeans_predictor.ipynb|Predict future price by using the files from kmeans.ipynb. Sample results are in "figures" folder.|
