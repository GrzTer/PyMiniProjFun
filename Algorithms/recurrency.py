"""
Book: Algorithms: Data Structures and Programming Structure - Piotr. Wróblewski
Book: Algorithms: illustrated guide - Adiya Y. Bhargava
Title: 2.9 Recurrency exercises
Date: 26.03.2025
Page: 47
"""

# Ex. 2-1 - Recurrency reverse list of integers
"""
def swap(lst, a, b):
    # Swap elements at indices a and b in lst.
    lst[a], lst[b] = lst[b], lst[a]

def reverse(lst, left, right):
    # Recursively reverse the list between indices left and right.
    if left < right:
        swap(lst, left, right)
        reverse(lst, left + 1, right - 1)

def main():
    lst = [1, 2, 3, 4, 5, 6, 7, 8]
    reverse(lst, 0, len(lst) - 1)
    for item in lst:
        print(item)

if __name__ == "__main__":
    main()
"""

# Ex. 2-2 - Recursive Binary Search - not commented
"""
def binary_search(lst, element_to_find, left, right):
    if left < right:
        mid = (left + right) // 2
        if lst[mid] == element_to_find:
            return mid
        elif element_to_find < lst[mid]:
            return f"{binary_search(lst, element_to_find, left, mid - 1)}"
        else:
            return f"{binary_search(lst, element_to_find, mid + 1, right)}"
    else:
        return -1

def main():
    lst = [1, 2, 3, 4, 5, 6, 7, 8]
    print(binary_search(lst, 3, 0, len(lst)))

if __name__ == "__main__":
    main()
"""

# Ex. 2-3-NormalRec - Int -> Bin

"""def int_bin_konwerter(x):
    if x != 0:
        int_bin_konwerter(x // 2)
        print(x % 2)

def main():
    int_bin_konwerter(100)

if __name__ == "__main__":
    main()"""

# Ex. 2-3-AdditionalNum - Int -> Bin
"""
def int_bin_konwerter(x):
    while x != 0:
        x = x // 2
        print(x % 2)
int_bin_konwerter(100)"""

# Finum
"""
def FiNum(n):
    if n < 2:
        return 1
    else:
        return FiNum(n - 1) + FiNum(n - 2)
print(FiNum(5))"""

# Fun
"""
def Fun(n):
    if n < 2:
        return 1
    else:
        return Fun(n - 1) + Fun(n - 2)
print(Fun(5))"""
"""
import sys
sys.setrecursionlimit(100000000)  # Increase the recursion limit to allow more recursive calls

def countdown(n):
    print(n)
    countdown(n - 1)

countdown(900)
"""
# P2.43) - Recurrency Greetings
"""
class Greetings:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print (f"Cześć, {self.name}!")
        self.greet2()
        print(f"Przygotowanie pożegnania...")
        self.bye()

    def greet2(self):
        print("Jak się masz, " + self.name + "?")

    def bye(self):
        print("Do zobaczenia!")

g = Greetings("Piotr")
g.greet()
"""

# P2.45 - Recurrency Fact
import sys

# sys.setrecursionlimit(100000000)
sys.set_int_max_str_digits(100000000)

"""
class Stack:
    def fact(self, x):
        if x == 1:
            return 1
        else:
            return x * self.fact(x - 1)
x = int(input("Podaj liczbę: "))
s = Stack()
print(s.fact(x))
"""


##### FreeCodeCamp
# FactorialNumber
"""
def factorial_number(x):
    if x == 1:
        return x
    else:
        # temp = factorial_number(x-1)
        temp *= x
    return temp
print(factorial_number(5))
"""
"""
def factorial_number(x):
    if x == 1: return x
    return x * factorial_number(x-1)
print(factorial_number(5))

"""


# Permutation - recuration
 """
def permutation_rec(string: str, pocket="") -> str:
    if len(string) == 0:
        print(pocket)
    else:
        for s in range(len(string)):
            letter = string[s]
            front = string[0:s]
            back = string[s+1:]
            together = front + back
            permutation_rec(together, letter + pocket)
print(permutation_rec("ABCDE", ""))
 """