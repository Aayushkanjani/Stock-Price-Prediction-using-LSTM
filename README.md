# Stock Price Prediction using LSTM

This project aims to predict stock prices with higher accuracy using the Long Short-Term Memory (LSTM). I have fetched real-time data from the `yfinance` library and employed Mean Absolute Error (MAE) as the metric and Adam optimizer for training the model.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Introduction

Stock price prediction is a critical task in the financial industry, with applications ranging from trading strategies to risk management. Accurate predictions can lead to significant financial gains, while poor predictions can result in substantial losses. Traditional methods of stock price prediction often rely on statistical models and technical analysis, which may not fully capture the complexities of market behavior.

In recent years, machine learning and deep learning techniques have shown great promise in improving the accuracy of stock price predictions. Among these techniques, Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), have proven particularly effective for time-series forecasting tasks due to their ability to learn and remember long-term dependencies.

In this project, I have utilized LSTM networks to predict future stock prices based on historical data. Leveraging the `yfinance` library to fetch real-time stock data, ensuring that our model is trained on up-to-date information. The LSTM model is trained using the Adam optimizer, which is known for its efficiency and effectiveness in training deep learning models. Evaluation of performance of the model has been done using Mean Absolute Error (MAE), a widely used metric for regression tasks that measures the average magnitude of errors between predicted and actual values.

By the end of this project, my aim is to provide a robust and accurate model for stock price prediction that can be used by traders, investors, and financial analysts to make informed decisions. This project also serves as an example of how advanced machine learning techniques can be applied to real-world financial data to solve complex problems.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook (optional, for interactive development)

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/stock-price-prediction-lstm.git
    cd stock-price-prediction-lstm
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Fetch and preprocess the data:
    ```python
    import yfinance as yf
    import pandas as pd

    def load_data(ticker, start, end):
        data = yf.download(ticker, start, end)
        data.reset_index(inplace=True)
        return data

    START = "2018-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    data = load_data('AAPL', START, TODAY)
    data.to_csv('apple_stock.csv')
    ```

2. Train the LSTM model:
    ```python
    # Load the data
    data = pd.read_csv('apple_stock.csv')

    # Preprocess the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    # Prepare the training and testing datasets
    train_data_len = int(np.ceil( len(data) * .70 ))
    train_data = scaled_data[0:int(train_data_len), :]

    # Create the training data set
    train_x, train_y = [], []
    for i in range(100, len(train_data)):
        train_x.append(train_data[i-100:i, 0])
        train_y.append(train_data[i, 0])

    train_x, train_y = np.array(train_x), np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0],train_x.shape[1],1))

    # Define and train the LSTM model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])
    model.fit(train_x, train_y, epochs=50, batch_size=32)

    # Save the model
    model.save('lstm_stock_model.h5')
    ```

3. Make predictions and plot the results:
    ```python
    from tensorflow.keras.models import load_model

    model = load_model('lstm_stock_model.h5')

    # Testing data set
    test_data = scaled_data[train_data_len - 100:, :]
    test_x, test_y = [], data['Close'][train_data_len:].values
    for i in range(100, len(test_data)):
        test_x.append(test_data[i-100:i, 0])

    test_x = np.array(test_x)
    test_x = np.reshape(test_x, (test_x.shape[0],test_x.shape[1],1))

    # Get the model's predicted price values 
    predictions = model.predict(test_x)
    predictions = scaler.inverse_transform(predictions)

    # Plot the data
    train = data[:train_data_len]
    valid = data[train_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

    # Model evaluation
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(test_y, predictions)
    print(f"Mean Absolute Error: {mae}")
    ```

## Model Architecture

The LSTM model consists of the following layers:

- Four LSTM layers with 50, 60, 80, and 120 units, respectively.
- Dropout layers with rates of 0.2, 0.3, 0.4, and 0.5 respectively to prevent overfitting.
- A Dense layer with a single unit to output the predicted stock price.

## Evaluation

The model is evaluated using the Mean Absolute Error (MAE) metric. The Adam optimizer is used for training the model. The performance of the model is visualized by comparing the predicted stock prices against the actual stock prices.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.


## Acknowledgements

- The [yfinance](https://pypi.org/project/yfinance/) library for providing real-time stock data.
- The [Keras](https://keras.io/) library for simplifying the creation and training of neural network models.

---

Feel free to customize this `README.md` file to better fit your project's specifics and your preferences.
