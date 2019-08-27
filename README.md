# kmeans-candlestick
An attempt to classify candlestick patterns of IDX stocks by using TensorFlow k-means, and predict the purchase return and risk in &lt;1 week. 

Technical analysis' underlying premise is future prices is predictable through (thus, a function of) previous prices. Combined with a desire to automate stock analysis, as well as previous learnings at [edX course on machine learning](https://courses.edx.org/courses/course-v1:ColumbiaX+CSMM.101x+3T2018/course/), become a starting point for me to figure its possibility. 

*(For the story TL;DR: It failed, violating the financial axiom: Past returns is bad predictor of future returns.)*
*Please note that this was a personal side project, and thus it is not optimized for collaborative work.*

## The learning journey
Python & machine learning fundamentals proofed not enough to even complete the testing properly. Here's the 2 months journey roughly looks like: Starting with overwhelming TensorFlow mid-level language, switching to Keras & setting its environment, speeding up training process by sampling the test-space, working out the buggy Nvidia CUDA driver, learning & applying new evaluation metric (AUC), and at last getting promising testing result - showing good AUC and low loss levels.

Then, I started field-testing, using actual latest price to test its reliability - to find the reliability isn't there. This misalignment leads me to retraining the model with more rigorous method. The key turns out to be the data splitting method. 

Previously, the data format are 20-days price, then **binned randomly** to training, validating, and testing data. Later, this proofs to be a mistake. I experimented splitting the data based on year: 1 last year for testing, 2 previous years to validate, and the earlier years to train. Shortly, the test data agrees with the field-testing - the predictor fails.

Then I went a step back and consider the fundamentals: stock prices is not reliably shaped by previous days, but by news, 'pom-pom', large institutional investor, etc. Concluding the method's failure to group candlesticks (and the pressure to get a real job), I ended the experiment here.

## The data journey: *pull, prepare, train, test, evaluate, improve, repeat*
### Pulling
Using the readily available Python library and code snippet, then tailored to automatically download the latest IDX data.
### Data prepping
Mainly utilizing numpy & panda, experimented with various data prep method: data cleaning, normalizing, setting the variables, and binning them to train-validate-test data.
### Train, test
Leaving my abused laptop for hours at night, then collecting its hardwork it done while I peacefully slept.
### Evaluate
Devise a way to plot the data, and figuring useful variables as lithmus test, which results will be used to improve the model
### Improve the method
The part where I'm actually thinking of & figuring better model
### Repeat
For 2 months

## What I learned
* The importance of correct & sound hypotheses, **then** start working
* Not slaving my laptop for this - now it's an old man at only 3.5 years old


## Appendices

###### Results
Result samples can be seen in the 'figures' folder.

###### Key scripts
|Files|Description|
|---|---|
|idx_dl_csv.py|Data downloader from Yahoo Finance. Need txt file containing the ticker names in bash command. (Sample: idx_stocks_list.txt)|
|price_preprocessor.py|Filter and preprocess data downloaded from Yahoo Finance. Also contain post processor which isn't moved out yet. (prediction_normalize() function)|
|kmeans.ipynb|For training cluster centers. Outputs cluster centers, RMSE and expected profit in csv files.|
|kmeans_predictor.ipynb|Predict future price by using the files from kmeans.ipynb. Sample results are in "figures" folder.|
