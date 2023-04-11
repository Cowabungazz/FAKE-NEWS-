# Fake News Detection using Texts & Titles with Machine Learning-
This project aims to detect fake news using machine learning algorithms by analyzing the texts and titles of news articles. The dataset contains articles labeled as fake or real, which are preprocessed and analyzed using various machine learning models. The performance of the models is then compared, and the best model is selected for detecting fake news.

![image](https://user-images.githubusercontent.com/107359897/231129271-b6cbb73b-a362-42e8-a29b-d0397d568b0b.png)

## Data Cleaning
The raw data is imported from two CSV files, 'Fake.csv' and 'True.csv'. The data is cleaned by:
  - Combining the two dataframes
  - Adding labels to the dataframes
  - Shuffling the dataset
  - Removing special characters, digits, and stopwords from the 'title' and 'text' columns
  - Stemming and tokenizing the text
  - Stripping HTML tags
  - Removing text between square brackets
  - Lemmatizing words
  - Removing URLs
  - Converting the date column to datetime format
The cleaned dataset is saved as 'cleaned_news_data.csv'.

## Exploratory Analysis and Analytic Visualization
Various insights were derived from the cleaned data, including:
  - The dataset is balanced
  - Distribution of subjects varies between fake and real news, hence not used for modeling
  - Word clouds and most common words in 'title_clean' and 'text_clean' show the prominence of politics and current events in news headlines
  - Fake news titles tend to have longer character and word counts compared to real news titles
  - The average word length in both fake and real news titles is similar
  - Unigram, bigram, and trigram analysis reveal similar patterns in fake and real news articles
  - There has been a recent increase in fake news and a decrease in real news
  
## Machine Learning Models
### Decision Tree, Logistic Regression, Random Forest, Multinomial Naive Bayes, and SVM models were used for detecting fake news.
The data was transformed using Bag of Words, TF-IDF, and GloVe.
While the decision tree model achieved high accuracy on this specific dataset, there are several potential issues related to using decision trees with TF-IDF transformed text data:
  - High dimensionality
  - Sparse data
  - Lack of feature interactions
  - Imbalanced class distribution
  - Interpretability
Thus we proceeded to use the following models 'Logistic Regression', 'Random Forest', 'Multinomial Naive Bayes', 'SVM' instead. Model performance was compared based on accuracy, precision, recall, and F1 score.

The Random Forest model with Bag of Words and TF-IDF input types performed the best, achieving the highest scores in accuracy, precision, recall, and F1 score.

## Model Optimization
Parameter tuning was performed using RandomizedSearchCV for Logistic Regression, Random Forest, Multinomial Naive Bayes, and SVM models.

## Contributions
### Practical Motivation, Problem Formulation: 
#### Bryan
### Data Cleaning, Exploratory Analysis, Machine Learning Models, Analytic Visualization, Model Optimization: 
#### Chang Zen

## References
Online:
Dataset Source: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
Scikit-learn documentation: https://scikit-learn.org/stable/
Text preprocessing with NLTK and BeautifulSoup: https://www.nltk.org/ and https://www.crummy.com/software/BeautifulSoup/bs4/doc/
News Prediction 1: https://www.reuters.com/article/us-usa-fiscal-idUSKBN1EP0LK
News Prediction 2: https://web.archive.org/web/20161115024211/http://wtoe5news.com/us-election/pope-francis-shocks-world-endorses-donald-trump-for-president-releases-statement/
Offline:
Textbook 1: Jurafsky, D., & Martin, J. H. (2019). Speech and Language Processing (3rd Edition). Stanford University.
Textbook 2: Raschka, S., & Mirjalili, V. (2017). Python Machine Learning (2nd Edition). Packt Publishing.
Textbook 3: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning: With Applications in R. Springer.
