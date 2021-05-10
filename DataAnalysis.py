# Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
from nltk import everygrams, word_tokenize
from nltk.stem.porter import *

outputfolder = './images/'

stemmer = PorterStemmer()

# Have a look at a few rows in data
df_data = pd.read_csv('./Youtube02-KatyPerry.csv')
df_data.head(5)

# Display shape of data.
print('display shape of data=' + str(df_data.shape))

# display data columns
df_data.info()

# Analyse dependant variable-'CLASS'
df_data.CLASS.value_counts()
ax = sns.countplot(x="CLASS", data=df_data, palette='Accent');

ax.set(ylabel="# of comments"
       , title='Barchart of # of comments belonging to both Classes');

plt.savefig(outputfolder + 'ClassBalance.png')

# Analyse Authors
print('number of unique authors who have commented=' + str(df_data.AUTHOR.nunique()))

# Next, we will group the data by Authors to get the name of authors who have commented the highest.
df_authors = df_data.groupby('AUTHOR').AUTHOR.value_counts().sort_index(ascending=False).sort_values(ascending=False)
df_authors.head(10)

# Analysing Comment
# Let us check how many unique comments are there.
print('number of unique comments=' + str(df_data.CONTENT.nunique()))

# There are only  3 duplicate comments in the dataset.
# These 3 duplicate comments belong to class 0 and are created by different Author.

df_content = df_data.groupby('CONTENT').CONTENT.value_counts().sort_index(ascending=False).sort_values(ascending=False)
df_content.head(4)

# Analyzing Comment Text
df_data['CONTENT_LENGTH'] = df_data['CONTENT'].str.len()
df_data['CONTENT_LENGTH'].describe()

# From the above details we can see that 75% of the comments have a
# length of 116 characters. Now, let us see the histogram of comment lengths.
df_data.hist(column='CONTENT_LENGTH');
plt.title('histogram of CONTENT length');
plt.savefig(outputfolder + 'ContentHist.png')

# From the above histogram we can see that the histogram is left skewed.
# There are only two comments that are long ( more than 1000 characters).
df_data[df_data.CONTENT_LENGTH > 1000][['CLASS', 'AUTHOR', 'CONTENT']]

# We can see that these long comments both belong to class1.
# Let us now check if there is any difference in Content Length between the 2 Classes.
plt.style.use('seaborn-deep')
plt.figure(figsize=(10, 10))
data1 = df_data[df_data.CLASS == 1].CONTENT_LENGTH
data2 = df_data[df_data.CLASS == 0].CONTENT_LENGTH

bins = np.linspace(0, 800, 20)  # set the number of bins for Histogram

plt.hist(data1, bins, alpha=0.5, label='SPAM')
plt.hist(data2, bins, alpha=0.5, label='NOT SPAM')
plt.title('histogram of Comment length by CLass');
plt.legend(loc='upper right')
plt.savefig(outputfolder + 'ContentClassHist.png')

data1.describe()  # printing the distribution of SPAM Content
data2.describe()  # printing the distribution of NOTSPAM Content

# printing median of content length of both classes
print('median of length of SPAM Content=' + str(data1.median()) +
      ', median of length of NOT SPAM Content=' + str(data2.median()))


# function that accepts a regx pattern and removes it from the input string
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


# remove URL
df_data['tidy_text'] = df_data['CONTENT'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

# remove special chars
df_data['tidy_text'] = df_data['tidy_text'].str.replace('[^a-zA-Z#]', ' ')
df_data.head()

# Tokenizing
tokenized_text = df_data['tidy_text'].apply(lambda x: x.split())
tokenized_text.head()

# Stemming
tokenized_text = tokenized_text.apply(lambda x: [stemmer.stem(i)
                                                 for i in x])
for i in range(len(tokenized_text)):
    tokenized_text[i] = ' '.join(tokenized_text[i])
df_data['tidy_text'] = tokenized_text

df_data['tidy_text'].head()

all_words = ' '.join([text for text in df_data['tidy_text']])

# print WordCloud of most frequent words
from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig(outputfolder + 'wordcloud.png')

# ### 9. Unigrams and Bigrams
# Import stopwords with nltk.
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

stop = stopwords.words('english')

# convert text to lowercase
df_data['content_lowercase'] = df_data['CONTENT'].str.lower()

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
df_data['content_without_stopwords'] = df_data['content_lowercase'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# create Unigrams for SPAM CLASS using CountVectorizer
word_vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(df_data[df_data.CLASS == 1].content_without_stopwords)
frequencies = sum(sparse_matrix).toarray()[0]
df_frequencies = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])

df_frequencies.reset_index(inplace=True)
df_frequencies_sorted = df_frequencies.sort_values(by=['frequency'], inplace=False, ascending=False)
df_frequencies_sorted = df_frequencies_sorted[0: 10]

sns.set_color_codes("pastel")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="frequency", y="index", data=df_frequencies_sorted,
            label="Total", color="b");

ax.set(ylabel=" Word",
       xlabel="Number of occurences", title='Commonly occurring words in SPAM Class');
plt.savefig(outputfolder + 'SpamUnigrams.png')

# create Unigrams for NOT SPAM CLASS using CountVectorizer
word_vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(df_data[df_data.CLASS == 0].content_without_stopwords)
frequencies = sum(sparse_matrix).toarray()[0]
df_frequencies = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])

df_frequencies.reset_index(inplace=True)
df_frequencies_sorted = df_frequencies.sort_values(by=['frequency'], inplace=False, ascending=False)
df_frequencies_sorted = df_frequencies_sorted[0: 10]

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="frequency", y="index", data=df_frequencies_sorted,
            label="Total", color="b");
ax.set(ylabel=" Word",
       xlabel="Number of occurences", title='Commonly occurring words in NOT SPAM Class');
plt.savefig(outputfolder + 'HamUnigrams.png')

# create Bigrams for SPAM CLASS using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

word_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(df_data[df_data.CLASS == 1].content_without_stopwords)
frequencies = sum(sparse_matrix).toarray()[0]
df_frequencies = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])

df_frequencies.reset_index(inplace=True)
df_frequencies_sorted = df_frequencies.sort_values(by=['frequency'], inplace=False, ascending=False)
df_frequencies_sorted = df_frequencies_sorted[0: 10]

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="frequency", y="index", data=df_frequencies_sorted,
            label="Total", palette='Accent');

ax.set(ylabel=" Word",
       xlabel="Number of occurences", title='Commonly occurring bigrams in SPAM Class');
plt.savefig(outputfolder + 'SpamBigrams.png')

# create Bigrams for NOT SPAM CLASS using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

word_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform(df_data[df_data.CLASS == 0].content_without_stopwords)
frequencies = sum(sparse_matrix).toarray()[0]
df_frequencies = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])

df_frequencies.reset_index(inplace=True)
df_frequencies_sorted = df_frequencies.sort_values(by=['frequency'], inplace=False, ascending=False)
df_frequencies_sorted = df_frequencies_sorted[0: 10]

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="frequency", y="index", data=df_frequencies_sorted,
            label="Total", palette='Accent');

ax.set(ylabel=" Word",
       xlabel="Number of occurences", title='Commonly occurring bigrams in NOT SPAM Class');
plt.savefig(outputfolder + 'HamBigrams.png')

# Analyzing Date
df_datesdata = pd.read_csv('./Youtube02-KatyPerry.csv', index_col=2, parse_dates=True)
df_datesdata.head(5)

# Add columns with year, month, and weekday name
df_datesdata['Year'] = df_datesdata.index.year
df_datesdata['Month'] = df_datesdata.index.month
df_datesdata['Weekday_Name'] = df_datesdata.index.day_name()
# Display a random sampling of 5 rows
df_datesdata.sample(5, random_state=0)

# Analyse number of Comments by weekday
ax = sns.catplot(kind="count", x="Weekday_Name", palette="ch:.25", aspect=1.5, data=df_datesdata,
                 order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']);

ax.set(ylabel="# of comments",
       title='Distribution of Comments by weekday');
plt.savefig(outputfolder + 'DateByWeek.png')