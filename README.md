# LSTM-tf-2.x-for-HAR-with-Accelerometer
This is probably the 1st 'tf2.x-version' LSTM (available on GitHub) implemented for Accelerometer data.
A classic 'tf1.x-version' LSTM can be found [here](https://github.com/ieeeWang/LSTM-Human-Activity-Recognition). You can spend the whole weekend to figure out how the network is built by using tf1.x, or spend several miniuts on my tf2.x code to understand everthing.

This project needs two-step procedures:

(step 1) to train a 'best-trained-model' for transfer learning on a target dataset (with labeling). 
IPython Notebook available [here](./LSTM_UCI2012Acc_TransferLearning.ipynb).

(step 2) to use the saved 'best-trained-model' to test a target dataset (without labeling).
IPython Notebook available [here](./imec_Acc_case_application.ipynb).


Before you run the notebook files, Notice:

(1) the 16 excel files (for target dataset) should be put under the folder 'files', they are too large to send by email.
(2) the UCI dataset (source dataset) will be send by using 'Wetransfer' through another email. After download it, put it under the project folder 'imec_case_Acc_LSTM_demo'.
I will add another py file for preparing this data pipleline soon.

Validation result on a open dataset (UCI2012Acc):
<p align="center">
    <img width="200%" src="results/conf_matrix_9Acc_50Hz_92.6.png" style="max-width:200%;"></a>
</p>
