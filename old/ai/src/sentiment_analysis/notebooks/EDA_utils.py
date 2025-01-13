import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import string

# nlp Utils
def data_tokens(data,lower_case=True,remove_stop_words=False,remove_punctuation=False):
    """
    Tokenize the input data, with optional lowercasing, stop word removal, and punctuation removal.

    Parameters:
    data (pd.Series): A pandas Series containing sentences (strings).
    lower_case (bool): Whether to convert tokens to lowercase. Default is True.
    remove_stop_words (bool): Whether to remove stop words from tokens. Default is False.
    remove_punctuation (bool): Whether to remove punctuation from tokens. Default is False.

    Returns:
    list: A list of strings (tokens).

    Raises:
    ValueError: If input data is not a pandas Series.

    Example:
    >>> data = pd.Series(["This is a sentence.", "This is another sentence."])
    >>> tokens = data_tokens(data, lower_case=True, remove_stop_words=True, remove_punctuation=True)
    >>> print(tokens)
    ['sentence', 'another', 'sentence']
    """

    # Ensure the input data is a pandas Series
    if not isinstance(data, pd.Series):
        raise ValueError("Input data should be a pandas Series")
    
    # Concatenate the data into a single string
    corpus=data.str.cat(sep=' ')

    # Tokenize the data
    tokens = word_tokenize(corpus)

    # Lowercase the tokens
    if lower_case:
        tokens = [word.lower() for word in tokens]

    if remove_stop_words:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

    # Remove punctuation if specified
    if remove_punctuation:
        tokens = [word for word in tokens if word not in string.punctuation]
        

    return tokens


def get_non_alpha_numeric_tokens(data):
    # Get Tokens from the data
    tokens = data_tokens(data,lower_case=True,remove_stop_words=True,remove_punctuation=False)

    non_alpha_numeric = set([word for word in tokens if not word.isalnum()])

    return non_alpha_numeric




# Word-Frequency Analysis
def word_freq_analysis(data,n_most_common=10,plot_title="",plot=True):
    '''
    Perform word frequency analysis on the input data.
    
    Parameters:
    data (pd.Series): A pandas Series containing sentences (strings).

    '''
    # Get Tokens from the data
    tokens = data_tokens(data,lower_case=True,remove_stop_words=True,remove_punctuation=True)

    fdist= FreqDist(tokens)
    most_common_words=fdist.most_common(n_most_common)

    # Generate word clouds
    text = ' '.join(tokens)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)


    # Convert to pandas DataFrame
    df_common_words = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    if plot:
        # Plotti`ng
        # Set up the subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plot the word frequency bar plot on the first subplot
        sns.set_style("whitegrid")
        sns.barplot(x='Frequency', y='Word', data=df_common_words, palette='viridis', ax=axes[0])
        axes[0].set_title(f'Top {n_most_common} Most Common Words')

        # Plot the word cloud on the second subplot
        axes[1].imshow(wordcloud, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title('Word Cloud')


        # Add a title to the overall plot
        plt.suptitle(plot_title)

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        plt.show()

        return most_common_words,wordcloud

# TF-IDF Analysis
# def tfidf_analysis(data, ngram_range=(1, 1)):
    # """
    # Perform TF-IDF analysis on the input data.

    # Parameters:
    # data (list or pd.Series): A list or pandas Series containing text documents.
    # ngram_range (tuple): The range of n-grams to consider. Default is (1, 1) for unigrams.
    # """


    # # Initialize TF-IDF vectorizer
    # tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    # # Fit and transform the data
    # tfidf_matrix = tfidf_vectorizer.fit_transform(data)

    # # Get feature names
    # feature_names = tfidf_vectorizer.get_feature_names_out()


    # print(tfidf_matrix)
    # print(feature_names)
    # print(tfidf_vectorizer)

    # return tfidf_matrix, feature_names, tfidf_vectorizer
def n_gram_analysis(data, n=2, k_most_common=10, plot_title="", plot=True, color='blue'):
    """
    Perform n-gram analysis on the input data.

    Parameters:
    data (pd.Series): A pandas Series containing sentences (strings).
    n (int): The size of the n-grams. Default is 2 for bigrams.
    k_most_common (int): The number of most common n-grams to plot.
    plot_title (str): Title for the plot.
    plot (bool): Whether to plot the data. If True, the plot will be displayed.
    color (str): Color for the plot bars.

    Returns:
    nltk.probability.FreqDist: A frequency distribution of n-grams.

    Example:
    >>> data = pd.Series(["This is a sample sentence.", "This is another example."])
    >>> ngram_freq_dist = n_gram_analysis(data, n=2, plot_title="Example Plot", color='green')
    >>> print(ngram_freq_dist)
    FreqDist({('this', 'is'): 2, ('is', 'a'): 1, ('a', 'sample'): 1, ('sample', 'sentence'): 1, ('is', 'another'): 1, ('another', 'example'): 1})
    """
    # Get tokens from the data
    tokens = data_tokens(data, lower_case=True, remove_stop_words=True, remove_punctuation=True)

    # Generate n-grams from the list of tokens
    ngrams_list = list(ngrams(tokens, n))

    # Calculate the frequency distribution of n-grams
    freq_dist = FreqDist(ngrams_list)

    # Top k_most_common n-grams
    ngram_labels, ngram_freqs = zip(*freq_dist.most_common(k_most_common))

    if plot:
        # Plot N-gram Frequencies
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(ngram_labels)), ngram_freqs, color=color, label="")
        plt.xlabel('N-gram')
        plt.ylabel('Frequency')
        plt.title(f'Top {k_most_common} {n}-grams for {plot_title}')
        plt.xticks(range(len(ngram_labels)), ngram_labels, rotation=45)
        plt.show()

    return freq_dist


# Lexical Diversity Analysis
def lexical_diversity(data):

    # Get Tokens from the data
    tokens = data_tokens(data,lower_case=True,remove_stop_words=True,remove_punctuation=True)

    return len(set(tokens)) / len(tokens)
