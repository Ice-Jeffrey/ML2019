from wordcloud import WordCloud
import matplotlib.pyplot as plt


def word_cloud(messages):
    word_str = ''
    for words in list(messages):
        word_str += ' '.join(words)
    cloud = WordCloud(width=512, height=512).generate(word_str)
    plt.figure(figsize=(10, 8), facecolor='k')
    plt.imshow(cloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
