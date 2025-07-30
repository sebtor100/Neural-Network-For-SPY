# Neural-Network-For-SPY
# ML-Enhanced Trading Analysis System

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

This project implements an advanced trading analysis system that combines a sophisticated deep learning model (LSTM with Attention) with traditional technical analysis (TA). The core philosophy is an **ML-first approach** where the neural network's signal dictates the final trading decision, while TA is used to intelligently adjust the confidence level in that decision.

The system is designed to be a robust, end-to-end solution: it handles data collection, feature engineering, model training with best-practice regularization, and generates a final, interpretable analysis report with rich visualizations.

---

## Example Output

The system produces a detailed text-based report and two insightful plots: a price chart with key levels and a training performance summary.

### Analysis Report & Price Chart

The final analysis provides a clear signal, a confidence score, and a breakdown of both the ML and traditional indicator readings.




### Training Performance

The training process is transparent, with plots for loss and accuracy to validate model performance.



---

## Key Features

-   **Deep Learning Core:** Utilizes a PyTorch-based **LSTM with a Multi-Head Attention** mechanism to capture complex temporal patterns in market data.
-   **Rich Feature Engineering:** Generates a set of 22 technical indicators and market features, including MACD, RSI, Bollinger Bands, Stochastics, ATR, and moving average slopes to provide a comprehensive view of the market.
-   **ML-First Decision Logic:** The final signal (Buy/Sell/Hold) is determined by the neural network. The confidence in this signal is then adjusted based on agreement or disagreement with a traditional, rule-based TA model.
    -   **Agreement:** Confidence is reinforced.
    -   **Disagreement:** Confidence is penalized, signaling an ambiguous market.
-   **Automated Training Pipeline:** Includes a full training workflow that fetches data for multiple tickers, handles class imbalance with `WeightedRandomSampler`, and uses **early stopping** to prevent overfitting, saving the best-performing model.
-   **Data-Driven Visualization:** Generates clean, publication-quality candlestick charts that automatically plot dynamically calculated support and resistance levels.

---

## System Architecture

The project is structured into modular classes, promoting clarity and extensibility.

1.  **Data Collection & Preparation (`TradingDataset`)**:
    -   Fetches historical price data using `yfinance`.
    -   Calculates all 22 technical indicators.
    -   Fits a `StandardScaler` on the combined training data to normalize features.
    -   Generates sequences of data (`[sequence_length, num_features]`) and corresponding future-return-based labels (Buy/Sell/Hold).

2.  **Neural Network (`TradingSignalNet`)**:
    -   An `nn.Module` containing an LSTM layer to process sequences, a Multi-Head Attention layer to focus on the most relevant time steps, and two separate output heads:
        -   **Classifier Head:** A multi-layer perceptron that outputs logits for the Buy/Sell/Hold classes.
        -   **Confidence Head:** A separate MLP with a Sigmoid activation to predict the model's own confidence.

3.  **Model Training (`TradingModelTrainer`)**:
    -   Orchestrates the training loop.
    -   Implements early stopping based on validation loss.
    -   Plots training history (loss and accuracy curves).

4.  **Main System (`EnhancedAdaptiveTradingSystem`)**:
    -   Integrates all components.
    -   The `generate_ml_enhanced_signal` method contains the core decision logic that prioritizes the ML signal and uses TA for confidence tuning.

---

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    Create a file named `requirements.txt` with the following content:
    ```
    yfinance
    pandas
    numpy
    matplotlib
    torch
    scikit-learn
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Use

The main execution block is located at the bottom of the script.

### 1. Train the Model from Scratch

To train the model on a pre-defined list of tickers and then analyze a target ticker (e.g., "SPY"):

-   Set `train_first=True`.
-   Run the script from your terminal.

```python
if __name__ == "__main__":
    # This will first train the model and save 'best_trading_model.pth'
    # before running the analysis on "SPY".
    run_analysis("SPY", train_first=True, epochs=50)


if __name__ == "__main__":
    # This will load the existing 'best_trading_model.pth' and run the
    # analysis directly on "TSLA".
    run_analysis("TSLA", train_first=False)
