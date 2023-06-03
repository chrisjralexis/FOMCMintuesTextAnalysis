import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup


def analyze_fomc_minutes(minutes_text):
    # Initialize the SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(minutes_text)

    # Analyze sentiment for each sentence
    sentiment_scores = []
    hawkish_count = 0
    dovish_count = 0

    for sentence in sentences:
        sentiment_score = sid.polarity_scores(sentence)['compound']
        sentiment_scores.append(sentiment_score)

        # Count hawkish and dovish words
        if sentiment_score > 0.1:
            hawkish_count += 1
        elif sentiment_score < -0.1:
            dovish_count += 1

    # Calculate the average sentiment score
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    # Classify as hawkish, dovish, or neutral based on sentiment score
    if average_sentiment > 0.1:
        interpretation = "Hawkish"
    elif average_sentiment < -0.1:
        interpretation = "Dovish"
    else:
        interpretation = "Neutral"

    # Calculate the percentage of hawkish and dovish words
    total_words = hawkish_count + dovish_count
    hawkish_percentage = (hawkish_count / total_words) * 100
    dovish_percentage = (dovish_count / total_words) * 100

    # Generate word cloud
    wordcloud = WordCloud().generate(minutes_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    return interpretation, hawkish_percentage, dovish_percentage


# HTML Web Scrape
url = 'https://www.federalreserve.gov/monetarypolicy/fomcminutes20230503.htm'

# Fetch the HTML content from the URL
response = requests.get(url)
html_content = response.text

# Extract the text from the HTML
soup = BeautifulSoup(html_content, 'html.parser')
minutes_text = soup.get_text()

interpretation, hawkish_percentage, dovish_percentage = analyze_fomc_minutes(minutes_text)
print("Interpretation:", interpretation)
print("Hawkish Words Percentage: {:.2f}%".format(hawkish_percentage))
print("Dovish Words Percentage: {:.2f}%".format(dovish_percentage))


