#!/bin/python3

import pandas as pd
from sklearn.metrics import accuracy_score


def predict(row):
    if row.Sex == 'female':
        return 1
    else:
        return 0


def main():
    """ main """
    # Wczytanie zbioru trenującego
    train = pd.read_csv('./train/train.tsv', sep='\t')
    # Pobranie nazw kolumn
    columns = train.columns


    # Wczytanie zbioru walidacyjnego
    dev_X = pd.read_csv('./dev-0/in.tsv', sep='\t', names=columns[1:], header=0)
    dev_y = pd.read_csv('./dev-0/expected.tsv', sep='\t', names=columns[:1], header=0)

    # Wykorzystanie funkcji predict do predykcji odpowiedzi
    dev_X['out'] = dev_X.apply(predict, axis=1)

    # Zapisanie odpowiedzi do pliku
    dev_X['out'].to_csv('./dev-0/out.tsv', sep='\t', index=False)

    # Wyświetlenie wyniki na zbiorze walidacyjnym
    print("ACCURACY DEV: {}".format(accuracy_score(dev_y, dev_X.out)))

    # Wczytanie zbioru testującego
    test_X = pd.read_csv('./dev-0/in.tsv', sep='\t', names=columns[1:], header=0)

    # Predykcja
    test_X['out'] = test_X.apply(predict, axis=1)

    # Zapis odpowiedzi do pliku
    test_X['out'].to_csv('./test-A/out.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()