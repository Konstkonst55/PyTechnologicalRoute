import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def test_learn():
    x = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0], ])  # данные для обучения

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans = kmeans.fit(x)

    test = np.array([[5, 1], [0, 3], [2, 1], [11, 1], [9, 3], [9, 1]])

    y = kmeans.predict(test)
    print('Predict: ', y)


def test_transform():  # it finally worked!!!
    data_frame = np.array(pd.read_csv("../data/dataset_details_osn_5k_col.csv", sep=";", encoding="utf-8"))

    data_frame = np.delete(data_frame, obj=0, axis=1)

    print(data_frame[:, 4])

    le = LabelEncoder()
    data_frame[:, 0] = le.fit_transform(data_frame[:, 0])
    data_frame[:, 4] = le.fit_transform(data_frame[:, 4])

    print(data_frame[:, 4])


def test_word2vec():
    sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
                 ['this', 'is', 'the', 'second', 'sentence'],
                 ['yet', 'another', 'sentence'],
                 ['one', 'more', 'sentence'],
                 ['and', 'the', 'final', 'sentence']]
    model = Word2Vec(sentences, min_count=1)
    print(model)


def test_encoder():
    # создаем двумерный массив с категориальными данными
    data = [['France'], ['Spain'], ['Germany'], ['Spain'], ['Germany'], ['France']]

    # создаем экземпляр класса LabelEncoder
    encoder = LabelEncoder()

    # применяем метод fit_transform к первому столбцу массива
    encoded_data = encoder.fit_transform([x[0] for x in data])

    # выводим закодированные данные
    print(encoded_data)


def test_array():  # ok!!
    arr2d3r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr2d1r = np.array([[1, 2, 3]])

    print(arr2d3r[:, 2])
    print(arr2d1r[:, 0])


def test_data_frame():
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=[1, 2, 3])
    print(df)
    nparr = np.array(df)
    print(nparr)


if __name__ == "__main__":
    test_data_frame()
