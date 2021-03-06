#!/usr/bin/env python
# coding: utf-8
# Задание 1 Структуры и алгоритмы.

# Условия

# Необходимо написать программу с функцией multiplicate(A), принимающей на вход 
# массив целых чисел А ненулевой длины и массив такой же длины, в котором на i-ом месте 
# находится произведение всех чисел массива А, кроме числа, стоящего на i-ом месте.

# Язык программирования: Python.
# Использование дополнительных библиотек и функций: не разрешается.
# В качестве решения необходимо прислать ссылку на GitHub.

# Пример

# На вход подается массив [1, 2, 3, 4]
# На выходе функции ожидается массив [24, 12, 8, 6]


def multiplicate(A: list) -> list:
    multiplicator = 1
    for i in A:
        multiplicator *=i
    
    rez_list = [int(multiplicator/i) for i in A]
    return rez_list


A = [1,2,3,4]
print(multiplicate(A))