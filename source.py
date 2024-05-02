import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
nltk.download('punkt')

middlebury_performance = {
    "2013" : 0.86,
    "2014" : 0.65,
    "2015" : 0.71,
    "2016" : 0.62,
    "2017" : 0.87,
    "2018" : 0.75,
    "2019" : 0.69,
    "2022" : 0.75,
    "2023" : 0.77,
    "2024" : 0.44,
    #"full" : 0.65,
}

williams_performance = {
    "2013" : 0.84,
    "2014" : 0.85,
    "2015" : 0.60,
    "2016" : 0.60,
    "2017" : 0.72,
    "2018" : 0.79,
    "2019" : 0.77,
    "2022" : 0.79,
    "2023" : 0.82,
    "2024" : 0.77,
    #"full" : 0.79,
}

amherst_performance = {
    "2013" : 0.94,
    "2014" : 0.87,
    "2015" : 0.72,
    "2016" : 0.81,
    "2017" : 0.68,
    "2018" : 0.65,
    "2019" : 0.83,
    "2022" : 0.63,
    "2023" : 0.42,
    "2024" : 0.58,
    #"full" : 0.54,
}

class DataPrep:

    def loadData(self):
        return pd.read_csv('nlp_project_data.csv')

    def cleanData(self, df):

        new_df = df

        def formatDate(date):
            return datetime.strptime(str(date.split(", ")[1]) + "-" + str(datetime.strptime(date.split(" ")[0], '%B').month) + "-" + str(date.split(", ")[0].split(" ")[1]), '%Y-%m-%d')

        new_df["Date"] = new_df.apply(lambda x: formatDate(x['Date']), axis=1)

        replacements = {
            '\n' : ' ',
            '\r' : ' ',
            '  ' : ' ',
            '.' : '',
            ',' : '',
            ':' : '',
            ';' : '',
            '(' : '',
            ')' : '',
            '[' : '',
            ']' : '',
            '{' : '',
            '}' : '',
            '!' : '',
            '?' : '',
            '&' : '',
            '*' : '',
            '%' : '',
            '#' : '',
        }

        def cleanText(text):
            new_text = text.lower()
            if (new_text.find("middlebury") != -1 or new_text.find("williams") != -1 or new_text.find("amherst") != -1):
                for key, value in replacements.items():
                    new_text = new_text.replace(key, value)
                new_text_arr = word_tokenize(new_text)

                return new_text_arr
            else:
                return np.nan



        new_df["Message"] = new_df.apply(lambda x: cleanText(x['Message']), axis=1)

        new_df.dropna(inplace=True)

        output = {
            "2013" : df[(df["Date"] >= datetime.strptime("2012-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2013-05-01", '%Y-%m-%d'))],
            "2014" : df[(df["Date"] >= datetime.strptime("2013-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2014-05-01", '%Y-%m-%d'))],
            "2015" : df[(df["Date"] >= datetime.strptime("2014-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2015-05-01", '%Y-%m-%d'))],
            "2016" : df[(df["Date"] >= datetime.strptime("2015-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2016-05-01", '%Y-%m-%d'))],
            "2017" : df[(df["Date"] >= datetime.strptime("2016-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2017-05-01", '%Y-%m-%d'))],
            "2018" : df[(df["Date"] >= datetime.strptime("2017-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2018-05-01", '%Y-%m-%d'))],
            "2019" : df[(df["Date"] >= datetime.strptime("2018-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2019-05-01", '%Y-%m-%d'))],
            "2022" : df[(df["Date"] >= datetime.strptime("2021-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2022-05-01", '%Y-%m-%d'))],
            "2023" : df[(df["Date"] >= datetime.strptime("2022-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2023-05-01", '%Y-%m-%d'))],
            "2024" : df[(df["Date"] >= datetime.strptime("2023-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2024-05-01", '%Y-%m-%d'))],
            "full" : df[(df["Date"] >= datetime.strptime("2020-05-01", '%Y-%m-%d')) & (df["Date"] < datetime.strptime("2024-05-01", '%Y-%m-%d'))]
        }

        return output
    

class Model:

    def trainModels(self, data):

        models = {
            "2013" : None, "2014" : None, "2015" : None, "2016" : None, "2017" : None, "2018" : None, "2019" : None,
            "2022" : None, "2023" : None, "2024" : None, "full" : None
            }

        for year in models.keys():
            df = data[year]
            models[year] = Word2Vec(sentences=df["Message"], vector_size=10, window=5, min_count=1, workers=4)

        return models


class BiasEval:

    def getCosineSimScores(self, teams, model, year):
        vader_lexicon = SentimentIntensityAnalyzer().lexicon

        model_vocab = list(model.wv.key_to_index.keys())

        pos_words = []
        neg_words = []

        for word in vader_lexicon.keys():

            if vader_lexicon[word] > 0 and word in model_vocab:
                pos_words.append(word)
            elif vader_lexicon[word] < 0 and word in model_vocab:
                neg_words.append(word)


        def cosineSim(w1, w2):
            return np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))

        def sentimentScore(team):

            team_emb = model.wv[team]

            pos_sims = []
            neg_sims = []
            std_sims = []

            for pos_word in pos_words:
                pos_sim = cosineSim(team_emb, model.wv[pos_word])
                pos_sims.append(pos_sim)
                std_sims.append(pos_sim)


            for neg_word in neg_words:
                neg_sim = cosineSim(team_emb, model.wv[neg_word])
                neg_sims.append(neg_sim)
                std_sims.append(neg_sim)

            return (np.mean(pos_sims) - np.mean(neg_sims)) / np.std(std_sims)
        

        sentimentScores = {}

        print("Sentiment Scores for " + year + ": ")
        for team in teams:
            score = sentimentScore(team)
            print("- "  + team + ": " + str(score))

            sentimentScores[team] = score

        print("\n")

        return sentimentScores
    
    def graphPerformance(self, scores):
        x = []
        y = []

        years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019" ,"2022", "2023", "2024"]

        color_arr = ["blue", "purple", "yellow"] * len(years)

        for year in years:
            x.append(scores[year]["middlebury"])
            y.append(middlebury_performance[year])

            x.append(scores[year]["amherst"])
            y.append(amherst_performance[year])

            x.append(scores[year]["williams"])
            y.append(williams_performance[year])

        plt.scatter(x, y, c=color_arr)

        plt.xlabel("Sentiment from D3 Boards")
        plt.ylabel("Performance of Team")

        plt.show()


