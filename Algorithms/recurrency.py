"""
Book: Algorithms: Data Structures and Programming Structure
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

def int_bin_konwerter(x):
    while x != 0:
        x = x // 2
        print(x % 2)
int_bin_konwerter(100)





