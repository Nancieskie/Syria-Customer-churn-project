# Syria-Customer-churn-project
CUSTOMER PREDICTION FOR SYRIATel
1. Business Understanding
Overview
Predicting and understanding customer churn is a critical aspect for any business. Customer churn not only affects the company's profitability but also entails higher costs to acquire new customers. This project uses machine learning algorithms to build a model that can accurately predict customers who will churn based on the information available in the dataset.
Challenges
The main challenges that the stakeholders at SyriaTel are faced with as a result of customer churn are:
1.	 Revenue Loss: Customer churn directly impacts a telco's revenue stream as it leads to the loss of subscription fees, service charges, and potential future sales.
2.	Decreased Market Share: High churn rates can erode a telco's market share, making it harder to compete in a crowded industry. As customers leave for competitors offering better deals or service quality, the telco's market position weakens, potentially leading to a downward spiral of further customer defections.
3.	Increased Acquisition Costs: Acquiring new customers to replace those lost to churn is expensive.
Proposed solution
To tackle the customer churn problem in telecommunication companies, our proposed solution leverages building a machine learning model that will identify at-risk customers. Through analysing our dataset about SyriaTel Customer Churn we will be able to build a classifier to predict whether a custom will soon exit. The results of this model will help the stakeholders who are the company owners know the factors that lead to customer churn and find ways of mitigating such risks.
Conclusion
In conclusion, our proposed solution for addressing customer churn in telecommunication companies revolves around leveraging machine learning to build predictive models that identify at-risk customers. By analyzing the SyriaTel Customer Churn dataset, we aim to develop a classifier capable of predicting customer exits. The insights derived from this model will empower stakeholders, including company owners, to understand the underlying factors driving churn and implement targeted mitigation strategies. Through this approach, we anticipate significant improvements in customer retention and overall business performance, ensuring sustained success in the competitive telecommunications market.
Problem statement
In the realm of telecommunication, customer churn poses a considerable challenge for companies (telcos), leading to the loss of subscribers who discontinue their services. Addressing this issue requires telcos to pinpoint customers at risk of churn and implement proactive strategies to retain them. Leveraging machine learning models becomes indispensable for telcos to predict potential churners, drawing insights from diverse data dimensions including customer usage behaviors, payment records, among many others.
Objectives
1.	Develop a Machine Learning Model: The primary objective is to develop a robust machine learning model capable of accurately identifying at-risk customers who are likely to churn based on the SyriaTel Customer Churn dataset.
2.	To identify the features that are important for predicting customer churn.
3.	Provide Actionable Insights: Extract actionable insights from the model predictions to help stakeholders, including company owners and decision-makers, understand the factors influencing customer churn and devise effective strategies to address these factors and improve customer retention rates.
2. Data Understanding
Sources:
The project uses a dataset from kaggle - SyriaTel Customer Churn dataset. The dataset contains various features related to customer behavior and account information, including whether the customer has churned or not. The dataset contains 21 columns and 3333 rows.
3. Data Preparation
Checking for missing values and duplicates
The dataset contains no missing values. There are no duplicates in the phone number entiries

4. Data Analysis
Univariate Analysis
"churn" is the target variable for this classification project.
The scaling differs across the features, and a few of the features are not normally distributed. The features will therefore have to be scaled and normalized.

5. Modeling
Random forest
Interpretation
Accuracy: The model correctly predicts the churn status for 96% of the customers. Precision for True (Churn): When the model predicts a customer will churn, it is correct 94% of the time. Recall for True (Churn): The model correctly identifies 70% of all actual churn cases. F1-score for True (Churn): The harmonic mean of precision and recall is 0.81, indicating a good balance between precision and recall.
Recommendations
Additional Models: Consider evaluating other models (e.g., Gradient Boosting) to see if they perform better.
Gradient Boosting
Interpretation: Accuracy (95.95%) The model correctly predicts the churn status of 95.95% of the customers in the test set. This indicates that the model is highly accurate overall.
Precision (88.61%) When the model predicts that a customer will churn, it is correct 88.61% of the time. This high precision indicates that there are relatively few false positives (i.e., customers predicted to churn who do not actually churn).
Recall (79.55%) The model correctly identifies 79.55% of the actual churn cases. This means that the model is able to detect a good proportion of customers who will actually churn, though it misses some (i.e., false negatives).
F1 Score (83.83%) The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both concerns. An F1 score of 83.83% indicates that the model has a good balance between precision and recall.
XGBoost Classifier
Interpretation:
Accuracy (95.95%): The model correctly predicts the churn status of 95.95% of the customers in the test set. This indicates that the model is highly accurate overall.
Precision (88.61%): When the model predicts that a customer will churn, it is correct 88.61% of the time. This high precision indicates that there are relatively few false positives (i.e., customers predicted to churn who do not actually churn).
Recall (79.55%): The model correctly identifies 79.55% of the actual churn cases. This means that the model is able to detect a good proportion of customers who will actually churn, though it misses some (i.e., false negatives).
F1 Score (83.83%): The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both concerns. An F1 score of 83.83% indicates that the model has a good balance between precision and recall.
Hyperparameter tuning the models
RandomForest Hyperparameter Tuning
The tuned Random Forest model improves its performance in predicting customer churn with high accuracy and precision. The recall rate, while lower than precision, is still reasonably high, suggesting that the model identifies a majority of the churn cases. The next step will be feature importance analysis to determine which features are most important for predicting churn. But first let us take a look at the results after tuning the other models
Hyperparameter Tuning for Gradient Boosting
Gradient Boosting with Simplified RandomizedSearchCV
Interpretation of Performance Metrics for the tuned model
Accuracy (96.40%): The model correctly predicts whether a customer will churn about 96.40% of the time. This high accuracy suggests that the Gradient Boosting model is very reliable in making predictions for this dataset.
Precision (88.10%): Out of all the customers the model predicted would churn, 88.10% actually did. This high precision indicates that the model is very good at identifying true churners, meaning fewer false positives.
Recall (84.09%): The model correctly identifies 84.09% of the actual churners. This is a strong recall value, meaning the model captures a large proportion of churners, though it misses about 16% of them.
F1 Score (86.05%): The F1 Score, which is the harmonic mean of precision and recall, is 86.05%. This balanced metric indicates that the model has a very good trade-off between precision and recall.
Hyperparameter Tuning for XGBoost
Interpretation of Performance Metrics for the tuned model
Accuracy (95.35%): The model correctly predicts whether a customer will churn about 95.35% of the time. This high accuracy suggests that the model is reliable in making predictions for this dataset.
Precision (88%): Out of all the customers the model predicted would churn, 88% actually did. This high precision indicates that the model is good at identifying true churners, meaning fewer false positives.
Recall (75%): The model correctly identifies 75% of the actual churners. While this is lower than precision, it means the model still captures a significant portion of the churners, though it misses 25% of them.
F1 Score (81%): The F1 Score, which is the harmonic mean of precision and recall, is 81%. This balanced metric indicates that the model has a good trade-off between precision and recall. Strategic Insights
Feature Importance Analysis with Random Forest
Based on the feature importance analysis for the models above, the top three features contributing to the prediction of customer churn are:
Total Day Minutes: It is possible that customers who use more minutes during the day may have higher expectations for service quality and network performance. If these expectations are not met, they might be more likely to churn.
Total Day Charge: It could be that higher charges might lead to dissatisfaction, especially if customers perceive the costs as too high relative to the value they receive. This dissatisfaction can increase the likelihood of churn.
Customer Service Calls: It is likely that customers who contact customer service frequently may be experiencing issues or dissatisfaction with the service. Repeated problems and potentially unsatisfactory resolutions can drive customers to switch to another provider.

6. Model Evaluation
To determine the best performing model, we need to compare their performance metrics after hyperparameter tuning.
Accuracy
Gradient Boosting: 96.40% Random Forest: 95.95% XGBoost: 95.35%
Precision
Random Forest: 91.78% Gradient Boosting: 88.10% XGBoost: 88.00%
Recall
Gradient Boosting: 84.09% Random Forest: 76.14% XGBoost: 75.00%
F1 Score:
Gradient Boosting: 86.05% Random Forest: 83.23% XGBoost: 80.98%
7. Conclusion
Gradient Boosting is the best-performing model based on the overall metrics.
8. Business Recommendations
Focus on Key Predictive Features
Total Day Minutes and Total Day Charge: Customers with high usage during the day are more likely to churn. Consider creating tailored plans with benefits for high day-time usage to retain these customers.
Customer Service Calls: High frequency of customer service interactions is a significant churn indicator. Improve customer service quality by:
o Enhancing training for service representatives.
o Reducing response times.
o Providing more efficient and satisfactory resolutions.
Targeted Retention Strategies
• Identify High-Risk Customers: Use the churn prediction model to identify customers at high risk of churning. Focus retention efforts on these customers.
• Personalized Offers: Provide personalized offers and discounts to high-risk customers based on their usage patterns and service interactions.
• Loyalty Programs: Implement loyalty programs to reward long-term customers and encourage their continued patronage.
Enhance Customer Engagement
• Regular Communication: Maintain regular communication with customers through newsletters, updates, and personalized messages.
• Value-Added Services: Introduce value-added services that enhance customer experience and provide additional benefits, such as entertainment subscriptions, priority customer service, etc.




