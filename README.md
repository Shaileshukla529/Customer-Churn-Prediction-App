# Customer Churn Prediction App

A Streamlit-based web application for predicting customer churn using machine learning models. This application helps businesses identify customers who are likely to churn (stop using their services) based on various customer attributes and service usage patterns.

## Features

- Interactive web interface built with Streamlit
- Real-time churn prediction
- Probability scores for predictions
- Input validation and error handling
- Support for multiple customer attributes

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shaileshukla529/Customer-Churn-Prediction-App.git
cd Customer-Churn-Prediction-App
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Fill in the customer details in the form:
   - Personal Information (Gender, Senior Citizen status)
   - Service Details (Phone Service, Internet Service, etc.)
   - Contract Information (Contract type, Payment method)
   - Usage Information (Tenure, Monthly Charges, Total Charges)

4. Click the "Predict Churn" button to get the prediction

## Model Details

The application uses a pre-trained machine learning model that has been trained on historical customer data. The model takes into account various factors including:
- Customer demographics
- Service usage patterns
- Contract details
- Payment history
- Service subscriptions

## Project Structure

```
Customer-Churn-Prediction-App/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── customer_churn_pipeline_lr.joblib    # Trained model pipeline
└── label_encoder_lr.joblib             # Label encoder for predictions
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit for the web framework
- scikit-learn for machine learning capabilities
- pandas for data manipulation 