#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.metrics import accuracy_score


def predict(row):
    """ Predykcja na podstawie płci pasażera """
    if row.Sex == 'female':
        return 1
    return 0


def main():
    """ main """
    # Wczytanie zbioru trenującego
    train = pd.read_csv('./train/train.tsv', sep='\t')
    # Pobranie nazw kolumn
    columns = train.columns


    # Wczytanie zbioru walidacyjnego
    dev_x = pd.read_csv('./dev-0/in.tsv', sep='\t', names=columns[1:], header=0)
    dev_y = pd.read_csv('./dev-0/expected.tsv', sep='\t', names=columns[:1], header=0)

    # Wykorzystanie funkcji predict do predykcji odpowiedzi
    dev_x['out'] = dev_x.apply(predict, axis=1)

    # Zapisanie odpowiedzi do pliku
    dev_x['out'].to_csv('./dev-0/out.tsv', sep='\t', index=False)

    # Wyświetlenie wyniki na zbiorze walidacyjnym
    print("ACCURACY DEV: {}".format(accuracy_score(dev_y, dev_x.out)))

    # Wczytanie zbioru testującego
    test_x = pd.read_csv('./dev-0/in.tsv', sep='\t', names=columns[1:], header=0)

    # Predykcja
    test_x['out'] = test_x.apply(predict, axis=1)

    # Zapis odpowiedzi do pliku
    test_x['out'].to_csv('./test-A/out.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
