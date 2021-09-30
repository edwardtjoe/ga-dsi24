### Overview

We conduct a Natural Language Processing on two subreddits, r/investing and r/personalfinance, and use several models to determine the origin of the post

### Problem Statement

Coach Ella Advisors requests the creation of a machine learning-based algorithm to identify the ways they can maximize the efficiency of their marketing spend based on their target audience.

### Background

Coach Ella Advisors is a newly created boutique financial advisory firm which provides advice to younger clients on personal financial health and planning. Although they have applied for an investment advisory license, it has yet to be approved. The founding team at Coach Ella believes that waiting for the approval of their investment advisory license would take too long and wishes to start marketing themselves immediately. As a newly created firm, they wish for us to minimize wasteful spending as they are unable to accept clients for or provide investment advice. However, they would also like for us to find the most efficient way to identify their potential clients as well.


### Datasets

We obtain data using PushShift API to collect data from two subreddits.

1. Pushshift API Link: https://github.com/pushshift/api
- Pushshift API allows us to collect data from reddit.com and collect processable data using json.

2. personalfinance.csv
- Subreddit for discussing personal financial health and planning
- Source: https://www.reddit.com/r/personalfinance/

3. investing.csv 
- Subreddit for discussing investing opportunities or methods
- Source: https://www.reddit.com/r/investing/

### Methodology

0. Imports
1. Data Collection and Cleaning
    - Collecting data from subreddits listed above
    - Merging datasets from the two subreddits
    - Using Regex to clean data
    - Tokenize post title and text
    - Process data using Lemmatizer, Porter Stemmer and Snow Stemmer
        - Remove stopwords
        - Remove subreddit names from all text
    - Dropping unnecessary columns
    - Determine stem to be used: Snow Stemmer
2. Exploratory Data Analysis and Modeling
    - Data Visualization
        - Word Cloud
        - Word Count
        - Ngram visualization
             - Unigram
             - Bigram
             - Trigram
    - Basic Model Creation
    - Create Function for Modeling
    - Logistic Regression CV (Count Vectorized, TF-IDF Vectorized)
    - Multinomial Naive Bayes (Count Vectorized, TF-IDF Vectorized)
    - Decision Tree (Count Vectorized, TF-IDF Vectorized)
    - Bootstrap Aggregating (Count Vectorized, TF-IDF Vectorized)
    - Random Forest (Count Vectorized, TF-IDF Vectorized)
    - Model Aggregation and Selection
3. Explanatory Data Analysis
    - Feature Importance
        - Top Key Words
        - Distribution of Key Word Strength
    - Misclassification Analysis
        - False Positive Analysis
        - False Negative Analysis
        - Word Similarity Analysis
4. Conclusion and Recommendation

### Results:

Model: Logistic Regression
Vectorizer: TF-IDF Vectorizer
Accuracy: 90.8%
Sensitivity: 91.6%
Specificity: 89.9%
Precision: 90.3%

### Conclusion

We conclude by stating our findings, and providing a recommendation based on our client's request.

In comparing our parameters, we find that a higher parameter number does not automatically translate to a better result. This is due to the diminishing returns in predictive power of the parameters. We also find that using an ngram range is better than using singular words only. This is because focus words often come in a phrase, similar to why supermarkets place soft drinks next to pizzas, or why casket-sellers also sell flowers.

In comparing our models, we find that a TF-IDF vectorized model provides a better predictive value than a Count Vectorized model. This is because while some stop words have been removed, there are still many stop words prevalent in our dataset. TF-IDF balances out the frequency of words used by penalizing common words which are not explicitly found in our stop words. This provides a higher weightage for non-stop words. Among the non-stop words, some of these are focus words used in the different subreddits due to their different focus. 

In comparing our two best models, we prefer using our logistic regression model as compared to our naive bayes model. Two reasons for this: maximization of our focus metric, and best overall balance in our 4 metrics. Our focus metric, specificity, performed best in a logistic regression. Due to the client's request of minimizing their incorrect spending, we desire the highest specificity in our models. In addition, our logistic regression scores over 90% on 3 out of 4 metrics, while specificity scores a hair below 90%, at 89.9%. Our naive bayes performed an impressive 95%+ on sensitivity, but poorly for specificity and precision (87.4% and 88.6% respectively). As such, while an equally weighted result points to using a naive bayes model (90.88% vs 90.65%), we believe that a logistic regression would be more reliable overall to meet the request of Coach Ella Advisory.
   
### Recommendation

1. Expand data collection from other sources (not just Reddit).
2. Identifying more stop words to reduce noise in our data.
3. Creating a dictionary to process words (stemming) more appropriately.
4. Obtain greater computing power or more time to process a greater number of hyperparameters.

