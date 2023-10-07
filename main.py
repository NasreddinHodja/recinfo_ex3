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


def generate_frequency_matrix(tokens_list, terms):
    frequency_matrix = pd.DataFrame(index=terms, columns=range(len(tokens_list)))
    for document, col in frequency_matrix.items():
        frequency_matrix[document] = col.index.to_series().apply(
            lambda term: tokens_list[document].count(term)
        )

    return frequency_matrix


def arrays_intersection(arrays):
    intersection = arrays[0]

    for arr in arrays[1:]:
        intersection = np.intersect1d(intersection, arr)

    return intersection


def arrays_union(arrays):
    union = arrays[0]

    for arr in arrays[1:]:
        union = np.union1d(union, arr)

    return union


def main():
    # input
    dictionary = np.array(
        [
            "O peã e o caval são pec de xadrez. O caval é o melhor do jog.",
            "A jog envolv a torr, o peã e o rei.",
            "O peã lac o boi",
            "Caval de rodei!",
            "Polic o jog no xadrez.",  # documentos
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
    print(terms)

    frequency_matrix = generate_frequency_matrix(tokens_list, terms)
    print(frequency_matrix)

    # # AND entre os termos da consulta
    # ands = arrays_intersection(simple_inverted_index)
    # print(f"AND = {ands}")

    # # OR entre os termos da consulta
    # ors = arrays_union(simple_inverted_index)
    # print(f"OR = {ors}")


if __name__ == "__main__":
    main()
