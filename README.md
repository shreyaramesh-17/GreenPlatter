# GreenPlatter 🥗  
### Sustainable Hotel Meal Management with AI  

GreenPlatter is an intelligent food waste reduction system that helps hotels optimize meal preparation using machine learning predictions.  
By analyzing historical data, weather patterns, events, and guest counts, it provides accurate demand forecasts to minimize food waste while ensuring customer satisfaction.  

## 🌟 Features  
-  AI-Powered Demand Prediction — Machine learning model trained on historical meal data  
-  Interactive Dashboard — Real-time analytics and waste tracking  
-  Smart Recommendations — Cooking quantity suggestions based on predicted demand  
-  Multi-Factor Analysis — Considers weather, events, day of week, and guest count  
-  Data Export — Download predictions and filtered data as CSV  
-  Waste Analytics — Track waste patterns by dish and time period

## 🛠 Tech Stack  
- Backend: Python, scikit-learn, pandas  
- Frontend: Streamlit  
- ML Model: HistGradientBoostingRegressor with cross-validation  
- Data Processing: One-hot encoding for categorical features  

## ⚡ Quick Start  

### ✅ Prerequisites  
bash
pip install streamlit pandas scikit-learn joblib
 
## 🚀 Setup  

1. Clone the repository
bash
git clone https://github.com/username/GreenPlatter.git
cd GreenPlatter

2. Place your dataset in the project folder
bash
Extended_GreenPlatter_12000.csv

3. Train the model
bash
jupyter notebook trial-1.ipynb


4. Run the application
bash
streamlit run app.py

## 📁 File Structure  
bash
GreenPlatter/
├── app.py                           # Streamlit application
├── trian.ipynb                      # Model training notebook
├── Extended_GreenPlatter_12000.csv  # Dataset (not included)
├── greenplatter_pipeline.joblib     # Trained model (generated)
├── greenplatter_categories.json     # Category metadata (generated)
└── README.md                        # Project documentation

## 📜 License  
This project is open source and available under the MIT License.  

## 🌍 Impact  

GreenPlatter helps hotels:  

- ♻ Reduce food waste by up to 30%  
- 📦 Optimize inventory management  
- 📊 Make data-driven cooking decisions  
- ✅ Track sustainability metrics  
- 💰 Improve cost efficiency

## 👩‍💻 Contributors
- Harshitha Siva (https://github.com/harshitha12uv)
- Shreya R (https://github.com/shreyaramesh-17)


### Built with ❤ for sustainable hospitality
