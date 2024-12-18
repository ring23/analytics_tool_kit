# Analytics Tool Kit - Streamlit App for Snowflae

## Description
This streamlit app provides the ability for users to perform the End-to-End ML Pipeline on any dataset in Snowflake. The user can perform EDA such as Correlation Heat Mapping, Missing Value/Outlier detection, descriptive stats, class imbalance, feature importance, ect. The user can then do some feature engineering, train a model, see model performance metrics and charts, run inference, and save the model to snowflake for future use.

---

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the App Locally](#running-the-app-locally)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

Before running the app locally, make sure you have the following installed:

- **Python 3.12.8** (Recommended version)
- **Streamlit** (You can install it with `pip install streamlit`)
- Additional dependencies listed in the `requirements.txt` file

---

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/ring23/analytics_tool_kit.git

2. Navigate to the project directory
   ```bash
   cd analytics_tool_kit

3. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate

4. Install requirements from requirements.txt
   ```bash
   pip install -r requirements.txt

4. Create secrets.toml file in .streamlit/ directory(create a file named secrets.toml inside the ./streamlit directory.
In the file, include the following and replace with your details:
   ```bash
   [snowflake]
   user = "user"
   password = "password"
   account = "account"

5. Once you have installed the dependencies, you can run the app locally using Streamlit:
   - Make sure you're in the project directory where app.py is saved and virtual environment is activated
   - Run the Streamlit App
   - The app should open in your default web browser. If not, you can access it by navigating to http://localhost:8501 in your browser.
   
  ```bash
  streamlit run app.py







   
