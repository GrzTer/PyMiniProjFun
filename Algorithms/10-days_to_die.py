"""
Day: 1/10
Brakes_Count:
Goal_Complition: 15/31
Start_Data: 30/03/2025
End_Data: 10/04/2025
"""

### -------Ogólne------- ###
# Fibonacci – Rekursja
"""
def fibonacci_recursion(n: int) -> int:
    # Jeżeli wartość n jest mniejsza lub równa 1, zwracamy n zgodnie z definicją ciągu (dla n=0 i n=1).
    if n <= 1:
        return n
    else:
        # Obliczamy n-ty wyraz ciągu rekurencyjnie, sumując dwa poprzednie wyrazy.
        return fibonacci_recursion(n - 1) + fibonacci_recursion(n - 2)

print(fibonacci_recursion(6))  # Dla n=6 oczekiwany wynik: 8.
"""
# Fibonacci – Iteracja
"""
def fibonacci_iteration(n: int) -> int:
    # Jeśli n jest mniejsze lub równe 1, zwracamy n, co odpowiada definicji ciągu (pierwsze wyrazy: 0 i 1).
    if n <= 1:
        return n
    else:
        a, b = 0, 1  # Inicjujemy zmienne: 'a' jako pierwszy wyraz, 'b' jako drugi wyraz ciągu.
        # Pętla iteracyjnie oblicza kolejne wyrazy ciągu od indeksu 2 do n (włącznie).
        for i in range(2, n + 1):
            a, b = b, a + b  # Aktualizacja: 'a' przyjmuje wartość poprzedniego 'b', a 'b' staje się sumą poprzednich 'a' i 'b'.
        # Po zakończeniu pętli 'b' zawiera n-ty wyraz ciągu.
        return b

print(fibonacci_iteration(6))  # Dla n=6 oczekiwany wynik: 8.
"""

# Silnia – Rekursja
"""
def factorial_recursion(n: int) -> int:
    # Dla n mniejszego lub równego 1 (zgodnie z definicją: 0! = 1 oraz 1! = 1) zwracamy 1.
    if n <= 1:
        return 1
    else:
        # Obliczamy n! rekurencyjnie jako iloczyn n oraz factorial_recursion(n - 1).
        return n * factorial_recursion(n - 1)

print(factorial_recursion(5))  # Dla n=5 oczekiwany wynik: 120.
"""
# Silnia – Iteracja
"""
def factorial_iteration(n: int) -> int:
    # Jeśli n jest mniejsze lub równe 1, zwracamy 1 zgodnie z definicją (0! = 1, 1! = 1).
    if n <= 1:
        return 1
    m = n
    # Pętla iteracyjnie mnoży kolejne liczby od n-1 do 1, obliczając w ten sposób n!.
    for i in range(1, n):
        m *= (n - i)
    return m

print(factorial_iteration(6))  # Dla n=6 oczekiwany wynik: 720.
"""

# Int -> Bin - Rekurencja
"""
def int_bin_recurrency(n: int) -> None:
    if n != 0:
        int_bin_recurrency(n // 2)
        print(n % 2)
int_bin_recurrency(6)
"""
# Int -> Bin - Iteracja
"""
def int_bin_iteration(n: int) -> None:
    while n != 0:
        n //= 2
        print(n % 2)
int_bin_iteration(6)
"""

# Recurrency reverse list of integers
"""
def rev_list(lst: list, left: int, right: int) -> list:
    if left < right:
        lst[left], lst[right] = lst[right], lst[left]
        rev_list(lst, left + 1, right - 1)
lst = [1,2,3,4,5,6,7,8,9,0]
rev_list(lst, 0, len(lst)-1)
for item in lst:
    print(item)
"""

# NWD - iteracja
"""
def gdc_iteration(a: int, b: int) -> int:
    # Oblicza największy wspólny dzielnik (NWD) liczb a i b metodą iteracyjną
    # (algorytm Euklidesa).
    # 
    # Parametry:
    #     a (int): Pierwsza liczba.
    #     b (int): Druga liczba.
    # 
    # Zwraca:
    #     int: Największy wspólny dzielnik a i b.
    while b:
        a, b = b, a % b  # Przypisanie: b staje się nowym a, a reszta z dzielenia a przez b staje się nowym b.
    return a
"""
# NWD - rekurencja
'''
def gdc_recursive(a: int, b: int) -> int:
    # Oblicza największy wspólny dzielnik (NWD) liczb a i b metodą rekurencyjną
    # (algorytm Euklidesa).
    #
    # Parametry:
    #     a (int): Pierwsza liczba.
    #     b (int): Druga liczba.
    #
    # Zwraca:
    #     int: Największy wspólny dzielnik a i b.
    if b == 0:
        return a
    else:
        return gdc_recursive(b, a % b)

print(gdc_recursive(100, 24))
'''

# Int -> Roman
"""
def int_roman_iteration(n: int) -> int:
    # Konwertuje liczbę całkowitą na jej reprezentację w postaci zapisu rzymskiego.
    #
    # Parametry:
    #     num (int): Liczba, którą chcemy przekonwertować. Zakres: 1 - 3999.
    #
    # Zwraca:
    #     str: Reprezentacja rzymska podanej liczby.
    #
    # Metoda:
    #     Iteracyjnie przechodzimy przez uporządkowaną listę par (wartość, symbol).
    #     Dla każdej pary, dopóki num jest większe lub równe danej wartości,
    #     odejmujemy tę wartość od num i do wyniku dodajemy odpowiadający symbol.

    # Lista par: wartość oraz odpowiadający jej symbol rzymski, uporządkowana malejąco.
    roman_numerals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    ]

    result = ""  # Inicjalizacja pustego ciągu, w którym będziemy budować zapis rzymski.

    # Iterujemy przez każdą parę (wartość, symbol) z listy.
    for value, symbol in roman_numerals:
        # Dopóki bieżąca liczba jest większa lub równa wartości z pary,
        # odejmujemy tę wartość od num i dodajemy symbol do wyniku.
        while n >= value:
            result += symbol
            n -= value
            # Każde odjęcie oznacza "zużycie" danej wartości w zapisie rzymskim.

    return result

    # Przykładowe wywołanie:


print(int_roman_iteration(1994))  # Oczekiwany wynik: MCMXCIV
"""

# Int -> Roman - Rekurencja
"""
def int_roman_recursiom(n: int) -> int:
    # Konwertuje liczbę całkowitą na jej reprezentację w postaci zapisu rzymskiego metodą rekurencyjną.
    #
    # Parametry:
    #     num (int): Liczba do konwersji (zakres: 1 - 3999).
    #
    # Zwraca:
    #     str: Reprezentacja rzymska podanej liczby.
    #
    # Metoda:
    #     Rekurencyjnie wybieramy największą możliwą wartość z listy par (wartość, symbol),
    #     która mieści się w num, odejmujemy tę wartość od num i do wyniku dołączamy
    #     odpowiadający symbol. Proces powtarzamy, aż num stanie się równy 0.

    # Lista par: wartość i odpowiadający jej symbol rzymski, uporządkowane malejąco.
    roman_numerals = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    ]

    # Warunek zakończenia rekurencji: gdy liczba osiągnie 0, zwracamy pusty ciąg.
    if n == 0:
        return ""

    # Iterujemy przez listę par, aby znaleźć pierwszą parę, której wartość mieści się w num.
    for value, symbol in roman_numerals:
        if n >= value:
            # Zwracamy symbol tej pary i wywołujemy rekurencyjnie funkcję dla reszty (num - value).
            return symbol + int_roman_recursiom(n - value)

    # Przykładowe wywołanie:


print(int_roman_recursiom(1994))  # Oczekiwany wynik: MCMXCIV
"""

# Anagram - iteracja
"""
def groupAnagrams(strs: list[str]) -> list[list[str]]:
    listlist = defaultdict(list) # mappinf characterCount to list of Anagrams
    for s in strs:
        key = [0] * 26 # a ... z = 26
        for c in s:
            key[ord(c) - ord("a")] += 1

        listlist[tuple(key)].append(s)
    return listlist.values()
print(groupAnagrams(["act","pots","tops","cat","stop","hat"]))
print(groupAnagrams(["act","pots","tops","cat","stop"]))
"""

# Anagram - rekursja
"""
def groupAnagrams_recursive(strs: list[str]) -> list[list[str]]:
    # Grupuje słowa będące anagramami przy użyciu rekurencyjnego przetwarzania.
    #
    # Parametry:
    #     strs (list[str]): Lista słów do pogrupowania.
    #
    # Zwraca:
    #     list[list[str]]: Lista grup anagramów.
    #
    # Metoda:
    #     1. Używamy rekurencji do przejścia przez listę słów.
    #     2. Dla każdego słowa, rekurencyjnie budujemy klucz – listę 26 elementów,
    #        gdzie każdy indeks odpowiada liczbie wystąpień danej litery ('a' do 'z').
    #     3. W mapowaniu (defaultdict) przypisujemy słowa do grup według klucza.

    # Funkcja rekurencyjna do obliczenia klucza dla słowa.
    def compute_key(s: str, index: int, key: list[int]) -> list[int]:
        if index == len(s):
            return key
        else:
            key[ord(s[index]) - ord("a")] += 1
            return compute_key(s, index + 1, key)

    # Funkcja rekurencyjna do przetworzenia listy słów.
    def process(index: int, mapping: dict) -> None:
        if index == len(strs): return
        # Wyznaczamy klucz dla aktualnego słowa rekurencyjnie.
        mapping[tuple(compute_key(strs[index], 0, [0] * 26))].append(strs[index])
        process(index + 1, mapping)
    mapping = defaultdict(list)
    process(0, mapping)
    return list(mapping.values())
# Przykładowe wywołania:
print(groupAnagrams_recursive(["act", "pots", "tops", "cat", "stop", "hat"]))
print(groupAnagrams_recursive(["act", "pots", "tops", "cat", "stop"]))
"""

### -------Wyszukiwanie------- ###

# Binarne wyszukiwanie(BS) - rekursja
"""
def bs_recursion(lst: list, element: int, low: int, high: int) -> int:
    if low <= high:
        mid = (low + high) // 2
        if lst[mid] == element: return mid
        elif lst[mid] > element: return bs_recursion(lst, element, low, mid - 1)
        else: return bs_recursion(lst, element, mid + 1, high)
    else: return None


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
print(bs_recursion(lst, 15, 0, len(lst)))
"""
# Binarne wyszukiwanie(BS) - iteracja
"""
def bs_iteration(lst: list, element: int) -> int:
    low = 0
    high = len(lst)
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] == element: return mid + 1
        elif lst[mid] > element: high = mid - 1
        else: low = mid + 1
    return None
print(bs_iteration([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 13))
"""

# ----------------------------------------------------------------------------------------------------------------------#
"""
Day: 2/10
Brakes_Count:
Goal_Complition: 15/31
Start_Data: 30/03/2025
End_Data: 10/04/2025
"""

### -------Ogólne------- ###

# sum-rekurencyjne
"""
def sun(lst: list) -> int:
    if not lst:
        return 0
    else:
        return lst[0] + sun(lst[1:])
print(sun([1,2,3]))
"""

#QuickSort(QS) - dziel i żądź
"""
def qs(lst:list)->list:
    if len(lst) < 2:
        return lst
    else:
        pivot = lst[0]
        lesser = [i for i in lst[1:] if i <= pivot]
        greater = [i for i in lst[1:] if i > pivot]
        return qs(lesser)+[pivot]+qs(greater)
print(qs([2,1,5,4,3]))
"""