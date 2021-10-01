import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Function to get the data - not used in this notebook but included for documentation purposes
def get_data(subreddit, df=None, has_df=False, cols_to_keep=['id', 'created_utc', 'author', 'title', 
                                                             'selftext', 'num_comments', 'score', 'upvote_ratio', 
                                                             'subreddit']):
    
    import datetime # For generating random dates
    import requests # For scraping data
    # Generate a random date (converted to utc) from May to August of 2020
    def gen_rand_date():
        # Define valid month/day/hour/minute/second values
        months = [5, 6, 7, 8]
        days = [i for i in range(1,32)]
        hours = [i for i in range(24)]
        minutes = [i for i in range(60)]
        seconds = [i for i in range(60)]
        
        # Randomly sample from each of the above values
        month = np.random.choice(months, size=1)[0]
        if month == 6:
            day = np.random.choice(days[:-1], size=1)[0] # June 2020 has 30 days
        else:
            day = np.random.choice(days, size=1)[0]
        hour = np.random.choice(hours, size=1)[0]
        minute = np.random.choice(minutes, size=1)[0]
        second = np.random.choice(seconds, size=1)[0]
        
        # Generate a random date (converted to utc timestamp)
        date = datetime.datetime(year=2020, month=month, day=day, hour=hour, minute=minute, second=second, 
                                tzinfo=datetime.timezone.utc).timestamp()
        
        # Return random utc timestamp
        return date
    
    def clean_df(df):
        df_copy = df.copy() # Avoid unwanted side-effects
        # Replace '' with NaN values from selftext columns
        df_copy.selftext.replace('', np.nan, inplace=True)
        # Replace [removed] with NaN values from selftext columns
        df_copy.selftext.replace('[removed]', np.nan, inplace=True)
        # Replace [deleted] with Nan values from selftext columns
        df_copy.selftext.replace('[deleted]', np.nan, inplace=True)
        # Drop rows with NaN values
        df_copy.dropna(subset=['selftext'], axis=0, inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        return df_copy
    
    def drop_dups(df):
        df_copy = df.copy() # Avoid unwanted side-effects
        df_copy.drop_duplicates(subset='id', keep='first', ignore_index=True, inplace=True)
        return df_copy
    
    def concat_data(df1, df2):
        final_df = pd.concat([df1, df2], axis=0)
        final_df.reset_index(drop=True, inplace=True)
        return final_df
    
    # Main function
    url = 'https://api.pushshift.io/reddit/search/submission'
    before = gen_rand_date()
    params = {
        'subreddit': subreddit,
        'size': 100,
        'pinned': False,
        'is_self': True,
        'user_removed': False,
        'mod_removed': False,
        'before': int(before)
        }
    
    if has_df == False:
        res = requests.get(url, params)
        posts = res.json()['data']
        ret_df = pd.DataFrame(posts)
        ret_df = drop_dups(ret_df)
        ret_df = clean_df(ret_df)
        return ret_df[cols_to_keep]
    
    else:
        res = requests.get(url, params)
        posts = res.json()['data']
        df2 = pd.DataFrame(posts)
        df2 = drop_dups(df2)
        df2 = clean_df(df2)
        ret_df = concat_data(df, df2)
        ret_df = drop_dups(ret_df) # On the off chance that df and df2 have duplicated entries
        return ret_df[cols_to_keep]

# Loads the data   
def load_data(pct=1):
    
    n_observations = int(pct * 10000)
    tomc_df = pd.read_csv('data/reddit_tomc.csv')
    unpopular_df = pd.read_csv('data/reddit_unpopular.csv')
    reddit_df = pd.concat([tomc_df.iloc[:n_observations, :], unpopular_df.iloc[:n_observations, :]], axis=0)
    reddit_df.drop(['Unnamed: 0', 'id', 'created_utc'], axis=1, inplace=True)
    reddit_df['y'] = reddit_df['subreddit'].apply(lambda s: 1 if s == 'TrueOffMyChest' else 0)
    return reddit_df
    
# Removes whitespace and new lines from dataframe
def clean_df(df):
    
    df_copy = df.copy()
    
    # Remove whitespace from title/selftext columns
    df_copy['title'].replace(r'\s', ' ', regex=True, inplace=True)
    df_copy['selftext'].replace(r'\s', ' ', regex=True, inplace=True)

    # Remove new lines from title/selftext columns
    df_copy['title'].replace(r'\n', ' ', regex=True, inplace=True)
    df_copy['selftext'].replace(r'\n', ' ', regex=True, inplace=True)
    
    return df_copy

# Visual check for imbalanced target variable
def check_balance(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='subreddit', data=df)
    plt.title('Number of Observations per Subreddit')

# Visualizes most common words
def visualize_most_common_words(df, subreddit, color, column='title', top_n=10):
    plt.figure(figsize=(10,6))
    vect = CountVectorizer(token_pattern=r"\b\w[\w']+\b", stop_words='english', max_features=500, 
                           strip_accents='unicode')
    titles = vect.fit_transform(df[df.subreddit == subreddit][column])
    title_df = pd.DataFrame(titles.toarray(), columns=vect.get_feature_names())
    title_df.sum().sort_values(ascending=False)[:top_n][::-1].plot(kind='barh', color=color)
    plt.title(f'Top {top_n} most common {column} words in the {subreddit} subreddit')
    
# Visualize word count distribution
def visualize_word_count_distribution(df, subreddit, color, column='title'):
    plt.figure(figsize=(10,6))
    df_copy = df.copy()
    df_copy = df_copy[df_copy.subreddit == subreddit]
    df_copy[column + '_' + 'word_count'] = df_copy[column].apply(lambda s: len(s.split()))
    sns.distplot(df_copy[column + '_' + 'word_count'], color=color)
    plt.title(f'{column} word count distribution of {subreddit} subreddit.')
    
# Hyperparameter tuning and model evaluation
def fit_predict_grid(how, model, params, n_grams, max_features=None, cv=5, randomized=False, n_iter=None):
    
    global X_train, X_test, y_train, y_test
    
    if how == 'bow': # Bag of words
        
        bow = CountVectorizer(ngram_range=n_grams, max_features=max_features, stop_words='english')
        
        if randomized: # Perform randomized grid search
            pipe = make_pipeline(bow, model)
            grid = RandomizedSearchCV(estimator=pipe, param_distributions=params, n_iter=n_iter, cv=cv, 
                                      random_state=207, n_jobs=-1)
            grid.fit(X_train, y_train)
            y_train_pred = grid.predict(X_train)
            y_test_pred = grid.predict(X_test)
            train_score = accuracy_score(y_train, y_train_pred)
            val_score = grid.best_score_
            test_score = accuracy_score(y_test, y_test_pred)
            return grid, train_score, val_score, test_score
    
        else: # Perform regular grid search
            pipe = make_pipeline(bow, model)
            grid = GridSearchCV(estimator=pipe, param_grid=params, cv=cv, n_jobs=-1)
            grid.fit(X_train, y_train)
            y_train_pred = grid.predict(X_train)
            y_test_pred = grid.predict(X_test)
            train_score = accuracy_score(y_train, y_train_pred)
            val_score = grid.best_score_
            test_score = accuracy_score(y_test, y_test_pred)
            return grid, train_score, val_score, test_score
        
    if how == 'tfidf': #TF-IDF
        
        tfidf = TfidfVectorizer(ngram_range=n_grams, max_features=max_features, stop_words='english')
        
        if randomized: # Perform randomized grid search
            pipe = make_pipeline(tfidf, model)
            grid = RandomizedSearchCV(estimator=pipe, param_distributions=params, n_iter=n_iter, cv=cv, 
                                      random_state=207, n_jobs=-1)
            grid.fit(X_train, y_train)
            y_train_pred = grid.predict(X_train)
            y_test_pred = grid.predict(X_test)
            train_score = accuracy_score(y_train, y_train_pred)
            val_score = grid.best_score_
            test_score = accuracy_score(y_test, y_test_pred)
            return grid, train_score, val_score, test_score
        
        else: # Perform regular grid search
            pipe = make_pipeline(tfidf, model)
            grid = GridSearchCV(estimator=pipe, param_grid=params, cv=cv, n_jobs=-1)
            grid.fit(X_train, y_train)
            y_train_pred = grid.predict(X_train)
            y_test_pred = grid.predict(X_test)
            train_score = accuracy_score(y_train, y_train_pred)
            val_score = grid.best_score_
            test_score = accuracy_score(y_test, y_test_pred)
            return grid, train_score, val_score, test_score
        
    return None

# Get model performance results
def get_results_df():

    cols = ['train_score', 'validation_score', 'test_score', 'kind']
    vals = [
        [train_score_lr_bow, train_score_nb_bow, train_score_rf_bow, train_score_boost_bow, 
         train_score_lr_tfidf, train_score_nb_tfidf, train_score_rf_tfidf, train_score_boost_tfidf],
        [val_score_lr_bow, val_score_nb_bow, val_score_rf_bow, val_score_boost_bow, val_score_lr_tfidf,
         val_score_nb_tfidf, val_score_rf_tfidf, val_score_boost_tfidf],
        [test_score_lr_bow, test_score_nb_bow, test_score_rf_bow, test_score_boost_bow, test_score_lr_tfidf,
         test_score_nb_tfidf, test_score_rf_tfidf, test_score_boost_tfidf],
        ['BoW', 'BoW', 'BoW', 'BoW', 'TF-IDF', 'TF-IDF', 'TF-IDF', 'TF-IDF']
    ]
    inds = ['logistic regression', 'naive bayes', 'random forest', 'boosting'] * 2
    ret_df = pd.DataFrame(dict(zip(cols, vals)), index=inds)
    ret_df.index.rename('Model', inplace=True)
    return ret_df