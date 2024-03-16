
# Indonesian Stock Forecasting With Realtime Evaluation

## Description
The Indonesian Stock Forecasting Project is an initiative to develop sophisticated prediction models for estimating stock price movements in the Indonesian stock market. Utilizing data taken in real-time from the official [IDX (Indonesia Stock Exchange)](https://www.idx.co.id/) website, this project aims to provide a reliable tool for investors to make more informed investment decisions.

## Features
- **Real-time Data Retrieval:** Stock data is directly taken from IDX to ensure the accuracy and relevance of the information.
- **Two Types of Models:** This project provides two types of models; Default and Tuned.
  - **Default Model:** This model is trained using the entire available dataset to give a general overview of market trends.
  - **Tuned Model:** With a more selective approach, this model is trained on only a specifically chosen subset of data. The aim is to enhance the reliability and accuracy of predictions through more meticulous parameter tuning.

## How to Use
1. **Environment Setup:**
    Ensure Python and all dependencies are installed. Clone this repository and run `pip install -r requirements.txt` to install required packages.
    
2. **Data Retrieval:**
    Stock data is taken in real-time from IDX. Use the provided script to download the latest data. Enter the IDX code of the company you wish to analyze in the specified format by the script.
    
    Example:
    ```
    python download_data.py --company_code "BBCA"
    ```
    Replace `"BBCA"` with the IDX code of the company you are interested in.

3. **Model Training:**
    - For the default model, run the training script with default parameters.
    - For the tuned model, adjust parameters as needed before running training.

4. **Evaluation and Prediction:**
    After training the model, use test data to evaluate model performance. Use the model to make predictions about stock price movements based on the entered company code.

## Contributions
We welcome contributions from the community! If you would like to contribute, please fork this repository, make your changes, and submit a pull request.

## License
This project is under the MIT license. See the `LICENSE` file for more information.
