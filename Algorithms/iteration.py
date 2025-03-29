"""
Book: Algorithms: Data Structures and Programming Structure - Piotr. Wróblewski
Book: Algorithms: illustrated guide - Adiya Y. Bhargava
Title: 1 Algorithm introduction
Date: 29.03.2025
Page: 9
"""
from numpy.ma.core import empty

# Binary search - iterative approach
"""
def BS(tab,n):
    low = 0
    high = len(tab)-1
    while low <= high:
        mid = (low+high)//2
        if tab[mid] == n:
            return mid
        elif tab[mid] > n:
            high= mid-1
        else:
            low= mid+1
    return None
print(BS([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 14))
print(BS([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], -1))
"""

# Finum - iteration
"""
def Finum(n):
    if n < 2:
        return 1
    a,b = 1,1
    for _ in range(2, n+1):
        a,b = b, a + b
    return b
print(Finum(10))
"""

# P2.45 - Iteration Fact
import sys

# sys.setrecursionlimit(100000000)
sys.set_int_max_str_digits(100000000)


class Factorial:
    def fact(self, x):
        result = 1
        for i in range(1, x + 1):
            result *= i
            print(result)
        return result


x = int(input("Podaj liczbę: "))
f = Factorial()
print(f.fact(x))
