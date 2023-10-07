#!/usr/bin/env python3

"""
Author: Tomás Bizet de Barros
DRE: 116183736
"""

import numpy as np
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


def generate_inverted_index(tokens_list, terms):
    inverted_index = []
    simple_inverted_index = []

    for term in terms:
        term_frequencies = []
        term_found = []
        for j, doc in enumerate(tokens_list):
            frequency = doc.count(term)

            if frequency:
                term_frequencies.append((j, frequency))
                term_found.append(j)

        inverted_index.append(np.array(term_frequencies))
        simple_inverted_index.append(np.array(term_found))

    return inverted_index, simple_inverted_index


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
    # # terms
    terms = np.array([term for l in tokens_list for term in l])

    inverted_index, simple_inverted_index = generate_inverted_index(tokens_list, terms)
    print(inverted_index)

    # AND entre os termos da consulta
    ands = arrays_intersection(simple_inverted_index)
    print(f"AND = {ands}")

    # OR entre os termos da consulta
    ors = arrays_union(simple_inverted_index)
    print(f"OR = {ors}")


if __name__ == "__main__":
    main()
