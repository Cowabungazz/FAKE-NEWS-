# Note: Pls refer to the zip project in the shared google drive folder to view the interactive plotly.io diagrams in our Jupyter files
https://drive.google.com/drive/folders/1xcpjzls9vZ82UabUL2D5sBA8gONdYsZd?usp=sharing


# Fake News Detection using Texts & Titles with Machine Learning- PROBLEM DEFINITION:
In the era of the internet and social media, the spread of misinformation and fake news has become a significant challenge for individuals, governments, and organizations worldwide. As the volume of digital content increases, it becomes increasingly difficult for consumers to differentiate between legitimate news sources and fake news. The consequences of this misinformation can be harmful, influencing public opinion, elections, and decision-making processes. This project aims to develop a comprehensive machine learning-based solution to identify and flag fake news articles, thereby enhancing the quality of information available to the public.

This project aims to detect fake news using machine learning algorithms by analyzing the texts and titles of news articles and to evaluate the performance of the model against a ground truth dataset. The dataset contains articles labeled as fake or real, which are preprocessed and analyzed using various machine learning models. The performance of the models is then compared, and the best model is selected for detecting fake news.

# Our Rationale:

- Mitigate the spread of misinformation: Identifying and flagging fake news will help reduce the spread of misinformation, ensuring that people have access to reliable and accurate information.

- Enhance public discourse: By reducing the influence of fake news, this project aims to promote a more informed and productive public discourse.

- Support fact-checking organizations: An automated fake news detection system can serve as a valuable tool for fact-checking organizations, enabling them to more efficiently and effectively identify and debunk false stories.

- Foster trust in news sources: By providing a means to detect fake news, this project aims to help restore public trust in legitimate news sources.


The following sections provide more details on the different steps and components of the project.

![image](https://user-images.githubusercontent.com/107359897/231129271-b6cbb73b-a362-42e8-a29b-d0397d568b0b.png)

## Data Cleaning
The data is preprocessed using Pandas and Natural Language Processing (NLP) libraries.
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
  - High dimensionality (detailed explanation provided in codebase)
  - Sparse data (detailed explanation provided in codebase)
  - Lack of feature interactions (detailed explanation provided in codebase)
  - Imbalanced class distribution (detailed explanation provided in codebase)
  - Interpretability (detailed explanation provided in codebase)

Thus we proceeded to use the following models 'Logistic Regression', 'Random Forest', 'Multinomial Naive Bayes', 'SVM' instead. Model performance was compared based on accuracy, precision, recall, and F1 score.

The best models for each input type were:
- Bag of Words: Random Forest (accuracy: 0.996612, precision: 0.996233, recall: 0.996702, F1 score: 0.996467)
- TF-IDF: Random Forest (accuracy: 0.996612, precision: 0.996700, recall: 0.996231, F1 score: 0.996466)
- GloVe: Random Forest (accuracy: 0.946917, precision: 0.951879, recall: 0.936631, F1 score: 0.944194)

Overall, the Random Forest model performed the best across all input types. Random Forest with Bag of Words or TF-IDF input types achieved accuracy and F1 scores above 0.996.

The Random Forest model with Bag of Words and TF-IDF input types performed the best, achieving the highest scores in accuracy, precision, recall, and F1 score.

## Model Optimization
Parameter tuning was performed using RandomizedSearchCV for Logistic Regression, Random Forest, Multinomial Naive Bayes, and SVM models.
The optimized models showed improved performance.

The best model showed to be "Bag of Words" input with optimised Random Forest:
- Accuracy: 1.00
- Precision: 1.00
- Recall: 1.00
- F1 Score: 1.00
- Best Parameters: {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': None}

## Prediction 
The best optimised model is saved and loaded for subsequent practical usage.

The optimised Random Forest model with Bag of Words successfully predicted random articles we googled online.
- "As U.S. budget fight looms, Republicans flip their fiscal script" -> TRUE
- "In 2016, a story circulated that Pope Francis made an unprecedented and shocking endorsement of Donald Trump for president" -> FALSE

## Contributions
### Practical Motivation, Problem Formulation: 
#### Bryan, Alex
### Data Cleaning, Exploratory Analysis, Analytic Visualization, Machine Learning Models, Model Optimization: 
#### Chang Zen
### Video Editing, Voice Over:
#### Bryan, Alex

## References
Online:
- Dataset Source: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Text preprocessing with NLTK and BeautifulSoup: https://www.nltk.org/ and https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- News Prediction 1: https://www.reuters.com/article/us-usa-fiscal-idUSKBN1EP0LK
- News Prediction 2: https://web.archive.org/web/20161115024211/http://wtoe5news.com/us-election/pope-francis-shocks-world-endorses-donald-trump-for-president-releases-statement/

Offline:
- Textbook 1: Jurafsky, D., & Martin, J. H. (2019). Speech and Language Processing (3rd Edition). Stanford University.
- Textbook 2: Raschka, S., & Mirjalili, V. (2017). Python Machine Learning (2nd Edition). Packt Publishing.
- Textbook 3: James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning: With Applications in R. Springer.
