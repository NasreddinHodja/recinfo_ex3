#!/usr/bin/env python3

"""
Author: Tomás Bizet de Barros
DRE: 116183736
"""

import numpy as np
import pandas as pd
import re


def tokenize(s, separators):
    pattern = "|".join(map(re.escape, separators))
    tokens = re.split(pattern, s)
    if tokens[-1] == "":
        tokens.pop()

    return np.array([token for token in tokens if token])


def normalize(s):
    normalized = s.lower().strip()
    return normalized


def remove_stopwords(tokens_list, stopwords):
    return [
        [token for token in tokens if token not in stopwords] for tokens in tokens_list
    ]


def generate_frequency_matrix(documents, terms):
    frequency_matrix = pd.DataFrame(index=terms, columns=range(len(documents)))
    for term, row in frequency_matrix.iterrows():
        frequency_matrix.loc[term] = row.index.to_series().apply(
            lambda doc: documents[doc].count(term)
        )

    return frequency_matrix


def and_all(frequency_matrix):
    return [
        index
        for index, value in enumerate(
            frequency_matrix.all(axis=0).map(lambda x: 1 if x else 0).to_list()
        )
        if value > 0
    ]


def or_all(frequency_matrix):
    max_frequencies = frequency_matrix.max()
    return [index for index, value in enumerate(max_frequencies) if value > 0]


def main():
    # input
    # documentos
    dictionary = np.array(
        [
            "O peã e o caval são pec de xadrez. O caval é o melhor do jog.",
            "A jog envolv a torr, o peã e o rei.",
            "O peã lac o boi",
            "Caval de rodei!",
            "Polic o jog no xadrez.",
        ]
    )
    stopwords = ["a", "o", "e", "é", "de", "do", "no", "são"]  # lista de stopwords
    query = "xadrez peã caval torr"  # consulta
    separators = [" ", ",", ".", "!", "?"]  # separadores para tokenizacao

    # normalize
    normalized = np.array([normalize(s) for s in dictionary])
    # tokenize
    tokens_list = np.array([tokenize(s, separators) for s in normalized], dtype=object)
    # rmv stopwords
    tokens_list = np.array(remove_stopwords(tokens_list, stopwords), dtype=object)
    # terms
    terms = np.array(list(set([term for l in tokens_list for term in l])))

    frequency_matrix = generate_frequency_matrix(tokens_list, terms)

    # # AND entre os termos da consulta
    ands = and_all(frequency_matrix)
    print(f"AND = {ands}")

    # OR entre os termos da consulta
    ors = or_all(frequency_matrix)
    print(f"OR = {ors}")


if __name__ == "__main__":
    main()
