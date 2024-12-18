# **Big Mart Sales Predictor**

## **Project Summary**
The Big Mart Sales Predictor is a machine learning-based web application designed to predict the sales of products across various outlets of a retail chain. By analyzing the product and outlet features, the app provides sales insights and predictions to help businesses optimize inventory management, improve sales strategies, and enhance profitability.  

## **Objectives**
1. Accurately predict the sales (`Item_Outlet_Sales`) for a given product based on its attributes and outlet information.  
2. Enable retail businesses to understand the relationship between product characteristics, outlet features, and sales performance.  
3. Provide an interactive and user-friendly interface for prediction and data insights.  
4. Visualize feature importance for better interpretability of the model.  

## **Methodology/Methods**
1. **Data Preprocessing:**
   - Cleaned and encoded categorical features like `Item_Fat_Content`, `Item_Type`, and `Outlet_Type`.  
   - Normalized numeric features like `Item_Weight`, `Item_Visibility`, and `Item_MRP` for better model performance.  
   - Applied logarithmic transformation to the target variable (`Item_Outlet_Sales`) to handle skewness.  

2. **Model Selection:**
   - Trained multiple regression models including Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.  
   - Gradient Boosting Regressor was selected as the best-performing model based on evaluation metrics such as R² score, MAE, RMSE, and MAPE.  

3. **Feature Importance Analysis:**
   - Determined the relative importance of each input feature using the Gradient Boosting model.  

4. **Web Application:**
   - Built using Streamlit to provide a highly interactive user interface for sales prediction.  

## **How to Use**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/big-mart-sales-predictor.git
   cd big-mart-sales-predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the trained Gradient Boosting model (`gradient_boosting_model.pkl`) is placed in the project directory.  
4. Run the application:
   ```bash
   streamlit run app.py
   ```
5. Open the app in your browser and input product and outlet features to predict sales.  

## **User Interface Screenshot**
![1](https://github.com/user-attachments/assets/adfea1d8-a5b1-4a26-a9d5-b34da6cf2dca)

![2](https://github.com/user-attachments/assets/06f04f53-c7dc-4901-8aff-a130a3a66df5)

## **Conclusion**
The Big Mart Sales Predictor provides a simple and efficient way for retail businesses to predict sales based on product and outlet features. By leveraging machine learning and interactive visualizations, this tool enables users to make data-driven decisions and optimize their operations.

## **Future Enhancements**
1. Expand the model to predict multiple KPIs such as profit margins and stock depletion rates.  
2. Add additional visualization tools for real-time analytics.  
3. Incorporate time-series forecasting for seasonal sales trends.  
4. Integrate APIs for direct data input and results exportation.  

## **Contact**
**Developer:** Hamza  
**Email:** your.email@example.com  
**GitHub:** [Your GitHub Profile](https://github.com/mrhamxo)  
**LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/in/muhammad-hamza-khattak/)  
