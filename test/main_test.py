from pathlib import Path

import joblib
import numpy as np
from gensim.models import Word2Vec
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


# файл для unit-тестирования, но пока просто тестирование разных фич

def __test_learn():  # ✅
    x = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0], ])  # данные для обучения

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans = kmeans.fit(x)

    test = np.array([[5, 1], [0, 3], [2, 1], [11, 1], [9, 3], [9, 1]])

    y = kmeans.predict(test)
    print('Predict: ', y)


def __test_transform():  # ✅
    data_frame = np.array([['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
                           ['this', 'is', 'the', 'second', 'sentence'],
                           ['yet', 'another', 'sentence'],
                           ['one', 'more', 'sentence'],
                           ['and', 'the', 'final', 'sentence']])

    data_frame = np.delete(data_frame, obj=0, axis=1)

    print(data_frame[:, 4])

    le = LabelEncoder()
    data_frame[:, 0] = le.fit_transform(data_frame[:, 0])
    data_frame[:, 4] = le.fit_transform(data_frame[:, 4])

    print(data_frame[:, 4])


def __test_word2vec():  # ✅
    sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
                 ['this', 'is', 'the', 'second', 'sentence'],
                 ['yet', 'another', 'sentence'],
                 ['one', 'more', 'sentence'],
                 ['and', 'the', 'final', 'sentence']]
    model = Word2Vec(sentences, min_count=1)
    print(model)


def __test_encoder():  # ✅
    # создаем двумерный массив с категориальными данными
    data = [['France'], ['Spain'], ['Germany'], ['Spain'], ['Germany'], ['France']]

    # создаем экземпляр класса LabelEncoder
    encoder = LabelEncoder()

    # применяем метод fit_transform к первому столбцу массива
    encoded_data = encoder.fit_transform([x[0] for x in data])

    # выводим закодированные данные
    print(encoded_data)


def __test_array():  # ✅
    arr2d3r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    arr2d1r = np.array([[1, 2, 3]])

    print(arr2d3r[:, 2])
    print(arr2d1r[:, 0])


def __test_data_frame():  # ✅
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=[1, 2, 3])
    print(df)
    nparr = np.array(df)
    print(nparr)


def __find_text_and_transform():  # ✅
    cols_name_list = ['name', 'gs_x', 'gs_y', 'gs_z', 'cg', 'mark', 'spf', 'tt']
    le_dict = {}
    df = DataFrame(
        [["Балка",
          261,
          0,
          0,
          90,
          "Д19ч",
          "Профиль",
          -1
          ],
         ["Палка",
          262,
          21,
          0,
          90,
          "Д20",
          "Профиль",
          "Шероховатость поверхности указана цветом в соответствии с инструкцией 30.0011.0155.998 . "
          "| Предельные отклонения размеров, допуски формы и расположения поверхностей - по ОСТ 1 00022-80 . "
          "| Контроль визуальный - после анодирования . "
          "| Покрытие: Ан.Окс.нхр Эмаль ЭП-140, оранжевая. 597 ОСТ 1 90055-85 . "
          "| Покрытие: Ан.Окс.нхр Эмаль ЭП-140, оранжевая. 597 ОСТ 1 90055-85 . "
          "| Покрытие: Ан.Окс.нхр Эмаль ЭП-140, оранжевая. 597 ОСТ 1 90055-85 . "
          "| Покрытие: Ан.Окс.нхр Эмаль ЭП-140, оранжевая. 597 ОСТ 1 90055-85 . "
          "| Маркировать Чк и клеймить Кк шрифтом ПО-5 ГОСТ 2930-62 . "
          "| Маркировать Чк и клеймить Кк шрифтом ПО-5 ГОСТ 2930-62 . "
          "| Маркировать Чк и клеймить Кк шрифтом ПО-5 ГОСТ 2930-62 . "
          "| Маркировать Чк и клеймить Кк шрифтом ПО-5 ГОСТ 2930-62 ."
          ]]
    )

    df.columns = cols_name_list

    for col in df.columns:
        first = df.loc[df.index[0], col]
        if isinstance(first, str) or first == -1:
            le_dict[col] = LabelEncoder()
            df[col] = le_dict[col].fit_transform(df.astype(str).__getattr__(col))

    print(df)
    print(le_dict['mark'].inverse_transform(df['mark']))


def __test_le():  # ✅
    df = DataFrame(
        [["1", "2", "3"],
         ["7", "2", "6"],
         ["7", "8", "6"]],
        columns=["1", "2", "3"]
    )
    df2 = DataFrame(
        [["7", "8", "6"]],
        columns=["1", "2", "3"]
    )
    le = {
        "1": LabelEncoder(),
        "2": LabelEncoder(),
        "3": LabelEncoder()
    }

    df["1"] = le["1"].fit_transform(df.astype(str).__getattr__("1"))
    df["2"] = le["2"].fit_transform(df.astype(str).__getattr__("2"))
    df["3"] = le["3"].fit_transform(df.astype(str).__getattr__("3"))
    print(df)
    joblib.dump(le, "../data/encoders/le.joblib")
    new_le = joblib.load("../data/encoders/le.joblib")
    print(Path("../data/encoders/le.joblib").exists())
    df2["1"] = new_le["1"].transform(df2.astype(str).__getattr__("1"))
    df2["2"] = new_le["2"].transform(df2.astype(str).__getattr__("2"))
    df2["3"] = new_le["3"].transform(df2.astype(str).__getattr__("3"))
    print(df2)
    print(le["1"].classes_)  # <----- ['1', '7']


if __name__ == "__main__":
    __test_le()
