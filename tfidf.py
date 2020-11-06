# -*- coding: utf-8 -*-
from typing import List, Set, Union
import math


class CountVectorizer:
    """Формирование словаря, подсчет количества вхождений по корпусу.
    :param encoding: кодировка
    :type encoding: str
    :param is_lower: приводить ли слова к нижнему регистру
    :type is_lower: bool
    """

    def __init__(self, encoding: str = 'utf-8', is_lower: bool = True):
        self.encoding = encoding
        self.is_lower = is_lower
        self.feature_names = set()
        self.vocabulary = {}

    def fit(self, corpus: List[str]):
        """Обучение, формирование уникальных слов, словаря.
        :param corpus: обучающий корпус
        :type corpus: List[str]
        """

        for text in corpus:
            for word in text.split(' '):
                if self.is_lower:
                    word = word.lower()
                self.feature_names.add(word)

        for index, word in enumerate(sorted(list(self.feature_names))):
            self.vocabulary[word] = index

    def transform(self, corpus: List[str]) -> List[List[int]]:
        """Преобразование входного корпуса в матрицу количества вхождений слов.
        :param corpus: входной корпус
        :type corpus: List[str]

        :rtype: List[List[int]]
        :return: матрица количества вхождений слов
        """

        count_matrix = [[0 for word in self.vocabulary] for text in corpus]
        for text_index, text in enumerate(corpus):
            counter = {}
            for word in text.split(' '):
                if self.is_lower:
                    word = word.lower()
                if word in self.vocabulary:
                    if word in counter:
                        counter[word] += 1
                    else:
                        counter[word] = 1
            for word, count in counter.items():
                word_index = self.vocabulary.get(word)
                if word_index is not None:
                    count_matrix[text_index][word_index] = count
        return count_matrix

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """Обучение и преобразование корпуса в матрицу количества вхождений слов.
        :param corpus: входной корпус
        :type corpus: List[str]

        :rtype: List[List[int]]
        :return: матрица количества вхождений слов
        """

        self.fit(corpus)
        return self.transform(corpus)

    def get_feature_names(self) -> Set[str]:
        """Возвращение уникальных слов обучающего корпуса.
        :rtype: Set[str]
        :return: уникальные слова обучающего корпуса
        """

        return self.feature_names


def tf_transform(
        count_matrix: List[List[int]],
) -> List[List[Union[float, int]]]:
    """Преобразование матрицы количества слов в tf слов
    :param count_matrix: входная матрица количества слов
    :type count_matrix: List[List[int]

    :rtype: List[List[Union[float, int]]]
    :return: матрица tf слов
    """

    return [
        [
            round(amount / sum(row), 3) for amount in row
        ] for row in count_matrix
    ]


def idf_transform(
        count_matrix: List[List[int]],
) -> List[Union[float, int]]:
    """Преобразование матрицы количества слов в idf слов
    :param count_matrix: входная матрица количества слов
    :type count_matrix: List[List[int]

    :rtype: List[Union[float, int]]
    :return: idf слов
    """

    len_matrix = len(count_matrix)
    words_occured = [
        [amount > 0 for amount in row] for row in count_matrix
    ]
    count_documents_with_word = [
        sum(word_occured) for word_occured in zip(*words_occured)
    ]
    return [
        round(
            math.log((len_matrix + 1) / (document + 1)) + 1, 3
        )
        for document in count_documents_with_word
    ]


class TfidfTransformer:
    """Превращает матрицу количества слов в tf-idf слов"""
    @staticmethod
    def fit_transform(
            count_matrix: List[List[int]],
    ) -> List[List[Union[float, int]]]:
        """Превращает матрицу количества слов в tf-idf слов
        :param count_matrix: входная матрица количества слов
        :type count_matrix: List[List[int]

        :rtype: List[List[Union[float, int]]]:
        :return: tf-idf слов
        """

        tf = tf_transform(count_matrix)
        idf = idf_transform(count_matrix)
        return [
            [
                round(
                    row_tf[row] * idf[row], 3
                ) for row in range(len(idf))
            ]
            for row_tf in tf
        ]


class TfidfVectorizer(CountVectorizer):
    """Превращение корпуса в tf-idf слов"""

    def fit_transform(
            self,
            corpus: List[str],
    ) -> List[List[Union[float, int]]]:
        """Превращение корпуса в tf-idf слов
        :param corpus: входной корпус
        :type corpus: List[str]

        :rtype: List[List[Union[float, int]]]:
        :return: tf-idf слов
        """

        count_matrix = super().fit_transform(corpus)
        tf_idf_matrix = TfidfTransformer().fit_transform(count_matrix)
        return tf_idf_matrix


if __name__ == "__main__":
    count_vectorizer = CountVectorizer()
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    count_matrix = count_vectorizer.fit_transform(corpus)

    print('feature_names: ', count_vectorizer.get_feature_names())
    print('count_matrix: ', count_matrix)

    tf_matrix = tf_transform(count_matrix)
    print('tf_matrix: ', tf_matrix)

    idf_matrix = idf_transform(count_matrix)
    print('idf_matrix: ', idf_matrix)

    tf_idf_transformer = TfidfTransformer()
    tf_idf_matrix = tf_idf_transformer.fit_transform(count_matrix)
    print('tf_idf_matrix: ', tf_idf_matrix)

    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_matrix = tf_idf_vectorizer.fit_transform(corpus)
    print('tf_idf_matrix: ', tf_idf_matrix)
