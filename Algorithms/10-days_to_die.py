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
Goal_Completion: 15/31
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

# QuickSort(QS) - dziel i żądź
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

# silnia
"""
def silnia(x:int)->int:
    if x <= 1:
        return x
    else:
        temp = 1
        for i in range (2, x+1):
            temp *= i
            print(temp)
        return temp
    return None
#silnia(5)

def silnia_rekurencja(x:int)->int:
    if x <= 1:
        return x
    else:
        return x * silnia_rekurencja(x-1)
    return None
print(silnia_rekurencja(6))
"""

# nwd
"""
def NWD(n: int, m: int) -> int:
    while m:
        n, m = m, n % m
    return n

#print(NWD(66, 6))

def NWD_rekurencja(n: int, m: int) -> int:
    if m == 0:
        return n
    else:
        return NWD_rekurencja(m, n % m)
    return None
print(NWD_rekurencja(24, 66))
"""

# ----------------------------------------------------------------------------------------------------------------------#
"""
Day: 3/10
Brakes_Count:
Goal_Completion: /31
Start_Data: 30/03/2025
End_Data: 10/04/2025
"""

# BS
"""
def BS(li: list, n: int) -> int:
    low, high = 0, len(li) -1
    while low <= high:
        mid = (low + high) // 2
        if li[mid] == n:
            return mid
        elif li[mid] > n:
            high = mid - 1
        else:
            low = mid + 1
    return None
print(BS([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 13)) # dla n=13 wynik(index)=12
"""
"""
def BS_r(li: list, n: int, low, high) -> int:
    if low <= high:
        mid = (low + high) // 2
        if li[mid] == n: return mid
        elif li[mid] > n: return BS_r(li, n, low, mid - 1)
        else: return BS_r(li, n, mid + 1, high)
    else:
        return None
li = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
print(BS_r(li, 13, 0, len(li) - 1)) # dla n=13 wynik(index)=12
"""

# fi
"""
def fi(x:int)->int:
    if x <= 1: return 1
    else:
        a, b = 0, 1
        for i in range(2, x + 1):
            a, b = b, a + b
        return b
    return None
print(fi(6))
"""
"""
def fi_r(n:int)->int:
    if n <= 1: return 1
    else: return fi_r(n - 1) + fi_r(n -  2)
    return None
print(fi_r(6))
"""

# silnia
"""
def fac(n:int)->int:
    if n <= 1: return 1
    else:
        a = 1
        for i in range(2, n + 1):
            a *= i
        return a
    return None
print(fac(5))
"""
"""
def fac_r(n:int) -> int:
    if n <= 1: return 1
    else: return n * fac_r(n - 1)
    return None
print(fac_r(6))
"""
"""
silnia = lambda x: 1 if x == 0 or x == 1 else x * silnia(x - 1)
print(silnia(5))
"""

# QS
"""
def qs_r(li: list) -> list:
    if len(li) < 2:
        return li
    else:
        # p = li[0]
        # l = [i for i in li[1:] if i <= li[0]]
        # g = [i for i in li[1:] if i > li[0]]
        # return qs_r(l + [p] + g)

        return qs_r([i for i in li[1:] if i <= li[0]]) + [li[0]] + qs_r([i for i in li[1:] if i > li[0]])
    return None


print(qs_r([2, 1, 5, 4, 3]))
"""

# BFS -------raz
"""
from collections import deque, defaultdict
graph = defaultdict(list)
graph["ty"] = ["alicja", "bartek", "cecylia"]
graph["bartek"] = ["janusz", "patrycja"]
graph["alicja"] = ["patrycja"]
graph["cecylia"] = ["tamara", "jarek"]
graph["janusz"] = []
graph["patrycja"] = []
graph["tamara"] = []
graph["jarek"] = ["adam"]
graph["adam"] = []

def bfs(graph: defaultdict) -> bool: # przyjmuje słownik i zwraca wartość boolowską
    search_queue = deque() # inicjuje kolejkę
    search_queue += graph["ty"] # dodaje do kolejki osoby 1 rzędowe
    searched = [] # inicjacja listy sprawdzonych osób

    def person_is_seller(name): return len(name) == 4 # funkcja zwracająca osobę sprzedającą mango

    while search_queue: # Dopóki kolejka nie jest pusta
        person = search_queue.popleft() # pobranie pierwszego pobranego elementu z kolejki
        if not person in searched:
            if person_is_seller(person): # warunek szukający sprzedawdcy mango
                print(f"{person} sprzedaje mango!") # wypisanie napisu np. "adam sprzedaje mango!"
                return True # zwrócenie wartości True i zakończenie programu
            else:
                search_queue += graph[person] # Przypadek, gdy osoba nie sprzedaje mango. Dodanie wszystkich znajomyc tej osoby do kolejki przeszukania
                searched.append(person)
    return False # zwracanie wartości False dla barku osoby w kolejce, która sprzedaje mango
print(bfs(graph)) # wywołanie funkcji z parametrem graph: defaultdict
"""
"""
graf = defaultdict(list)
graf["pobodka"] = ["cwiczenia", "mycie zebow", "spakowanie drugiego sniadania"]
graf["cwiczenia"] = ["prysznic"]
graf["prysznic"] = ["ubieranie sie"]
graf["mycie zebow"] = ["sniadanie"]
graf["prysznic"] = []

def bfs_cwiczenie(graf: defaultdict) -> bool:
    kolejka = deque()
    kolejka += graf["pobodka"]
    zrobione = []

    def czy_zrobione(czynnosc): return czynnosc[-1] == "w"

    while kolejka:
        czynnosc = kolejka.popleft()
        if not czynnosc in zrobione:
            if czy_zrobione(czynnosc):
                print(f"{czynnosc} wykonane! Możesz zjeść śniadanie")
                return True
            else:
                kolejka += graf[czynnosc]
                zrobione.append(czynnosc)
    return False
print(bfs_cwiczenie(graf))
# lista posortowana topologicznie:
#     1. pobodka
#     2. cwiczenia
#     3. mycie zebow
#     4. spakowanie drugioego sniadania
#     5. prysznic
#     6. sniadanie
#     7. ubieranie sie

#    ćwiczenia <- prysznic <- ubieranie się
#       /
#     pobodka <- mycie zębów <- śniadanie
#       /
#   spakowanie drugioego sniadania
"""

# Dijkstra
"""
graph = {}

graph["start"] = {}
graph["start"]["a"] = 6
graph["start"]["b"] = 2

graph["a"] = {}
graph["a"]["meta"] = 1

graph["b"] = {}
graph["b"]["a"] = 3
graph["b"]["meta"] = 5

graph["meta"] = {}

infinity = float("inf")
costs = {}
costs["a"] = 6
costs["b"] = 2
costs["meta"] = infinity

parents = {}
parents["a"] = "start"
parents["b"] = "start"
parents["meta"] = None

processed = []


def dijkstra(graph):
    def find_lowest_cost_node(costs):
        lowest_cost = infinity
        lowest_cost_node = None
        for node in costs: # Przegląda każdy węzeł po kolei
            cost = costs[node]
            if cost < lowest_cost and node not in processed: # Jeśli jest to najniższy z dotychczasowych kosztów i nie został jeszcze przetworzony...
                lowest_cost = cost # ...ustaw go jako nowy najtańszy węzeł
                lowest_cost_node = node
        return lowest_cost_node


    node = find_lowest_cost_node(costs) # Znajduje najtańszy węzeł, który nie został jeszcze przetworzony,
    while node is not None: # Jeśli wszystkie węzłu zostały przetworzone, następuje zakończenie pętli,
        cost = costs[node]
        neighbors = graph[node]
        for neighbor in neighbors.keys(): # Przegląda wszystkich sąsiadów danego węzła,
            new_cost = cost + neighbors[neighbor]
            if costs[neighbor] > new_cost: # Jeśli dotarcie do tego sąsiada jest tańsze drogą przez ten węzeł...
                costs[neighbor] = new_cost # ...zaktualizuj koszt tego węzła,
                parents[neighbor] = node # Węzeł ten staje się nowym rodzicem tego sąsiada,
        processed.append(node) # Oznaczenie węzła jako przetworzonego,
        node = find_lowest_cost_node(costs) # Znajduje następny węzeł do przetworzenia i wraca na początek pętli
    # Drukowanie wyników
    print("Koszty:", costs)
    print("Rodzice:", parents)
dijkstra(graph)
"""

# bs_i&bs_r
"""
def bs_i(l:list,n:int)->int:
    low, high = 0, len(l) -1
    while low <= high:
        mid = (low + high) // 2
        if l[mid] == n: return mid
        elif l[mid] < n: low = mid + 1
        else: high = mid - 1
    return None
print(bs_i([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9],-4))
"""
"""
def bs_r(lst:list,x:int,l:int,h:int)->int:
    if l<=h:
        m = (l+h) //2
        if lst[m] == x: return m
        elif lst[m] < x: return bs_r(lst,x,m+1,h)
        else: return bs_r(lst,x,l,m-1)
lst = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(bs_r(lst,-4, 0, len(lst)-1))
"""

# rev list of int recur
"""
def rev_lst_of_int_r(lst:list, left:int, right:int)->list:
    if left < right:
        lst[left], lst[right] = lst[right], lst[left]
        rev_lst_of_int_r(lst, left + 1, right - 1)
lst = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rev_lst_of_int_r(lst, 0, len(lst) - 1 )
for item in lst:
    print(item)
"""

# NWD-iter-recur
"""
def NWD_lub_GCD(a:int, b:int)->int:
    while b:
        a, b = b, a % b
    return a
print(NWD_lub_GCD(101234543248, 24))
"""
"""
def NWD_lub_GCD_r(a:int, b:int)->int:
    if b ==0: return a
    else: return NWD_lub_GCD_r(b, a % b)
print(NWD_lub_GCD_r(101234543248, 24))
"""
"""
def NWD_raz_jeszcze_r(a:int, b:int)->int:
    if b == 0: return a
    else: return NWD_raz_jeszcze_r(b, a % b)
print(NWD_raz_jeszcze_r(101234543248, 24))
"""
"""
def NWD_raz_jeszcze_i(a:int,b:int)->int:
    while b: a, b = b, a % b;
    return a
print(NWD_raz_jeszcze_i(101234543248, 24))
"""

# qs
"""
def qs(arr: list) -> list:
    if len(arr) < 2: return arr
    else: return qs([i for i in arr[1:] if i <= arr[0]]) + [arr[0]] + qs([i for i in arr[1:] if i > arr[0]])
print(qs([3,1,2,4,0,9,-10,-1,6,2,13,6,8,7]))
"""
"""
def qs(arr: list) -> list: #Zamiast listy to set, by pozbyć się powtórek
    arr = list(set(arr))  # Konwertowanie listy na zestaw i z powrotem na listę (usuwanie duplikatów)
    if len(arr) < 2: return arr
    else: return qs([i for i in arr[1:] if i <= arr[0]]) + [arr[0]] + qs([i for i in arr[1:] if i > arr[0]])

print(qs([3, 1, 2, 4, 0, 9, -10, -1, 6, 2, 13, 6, 8, 7]))
"""
"""
quick_sort = lambda arr: arr if len(arr) < 2 else quick_sort([i for i in arr[1:] if i <= arr[0]]) + [arr[0]] + [i for i in arr[1:] if i > arr[0]]
print(f"Posortowana lista - quicksort: {quick_sort([3, 1, 2, 4, 0, 9, -10, -1, 6, 2, 13, 6, 8, 7])}")
"""
# Bin
"""
def bin_konwert(x):
    while x:
        x //= 2
        print(x % 2)
    return None
bin_konwert(100)
"""

# Anagram
"""
def an(strs: list[str]) -> list[list[str]]:
    array = defaultdict(list) # Utworzyć słownik z defaultową wartością ustawioną na listę
    for s in strs: # Przejdź po każdym słowie z pobranej listy
        key = [0] * 26 #
        for w in s:
            key[ord(w) - ord("a")] += 1
        array[tuple(key)].append(s)
    return array.values()
print(an(["act", "pots", "tops", "cat", "stop", "hat"]))
"""

# ----------------------------------------------------------------------------------------------------------------------#
"""
Day: 4/10
Brakes_Count:
Goal_Completion: /31
Start_Data: 30/03/2025
End_Data: 10/04/2025
"""

# Algorytm zachłanny - Problem pokrycia zbioru - 146.p - Algorytmy :ilustrowany
"""
def radio_problem():
    states_needed = set(["mt", "wa", "or", "id", "nv", "ut", "ca", "az"])

    stations = {}
    stations["kone"] = set(["id", "nv", "ut"])
    stations["ktwo"] = set(["wa", "id", "mt"])
    stations["kthree"] = set(["or", "nv", "ca"])
    stations["kfour"] = set(["nv", "ut"])
    stations["kfive"] = set(["ca", "az"])

    final_stations = set()

    while states_needed:
        best_station = None
        states_covered = set()
        for station, states in stations.items():
            covered = states_needed & states
            if len(covered) > len(states_covered):
                best_station = station
                states_covered = covered
        states_needed -= states_covered
        final_stations.add(best_station)
    print(final_stations)
radio_problem()
"""

# qs
"""
def qs(arr:list)->list:
    print(arr)
    if len(arr) < 2: return arr
    else: return qs([i for i in arr[1:] if i <= arr[0]]) + [arr[0]] + qs([i for i in arr[1:] if i > arr[0]])

print(qs(["mt", "wa", "or", "id", "nv", "ut", "ca", "az"]))
"""

# bs
"""
def bs_i(arr: list, element: int) -> int:
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == element: return mid
        elif arr[mid] < element: low = mid + 1
        else: high = mid - 1
print(bs_i([-10, -1, 0, 1, 2, 2, 3, 4, 9, 6, 13, 6, 8, 7], 3))
"""

# Anagram
"""
def an(strs: list[str]) -> list[list[str]]:
    array = defaultdict(list)
    for st in strs:
        key = [0] * 26
        for s in st:
            key[ord(s) - ord("a")] += 1
        array[tuple(key)].append(st)
    return array.values()
print(an(["act", "pots", "tops", "cat", "stop", "hat"]))
"""

# ProblemPlecaka-Zachłanne
"""
def problem_plecaka_zachlannie():
    produkty = [{"nazwa": "gitara", "wartosc": 1500, "waga": 1},
                {"nazwa": "stereo", "wartosc": 3000, "waga": 4},
                {"nazwa": "laptop", "wartosc": 2000, "waga": 3}]
    
    
    def stosunek_wartosc_waga(produkt): return produkt["wartosc"] / produkt["waga"]
    
    
    max_waga = 4
    current_waga = 0
    current_wartosc = 0
    wybrane_produkty = []
    
    for produkt in produkty:
        if current_waga + produkt["waga"] <= max_waga:
            wybrane_produkty.append(produkt)
            current_waga += produkt["waga"]
            current_wartosc += produkt["wartosc"]
    
    print("Wybrane produkty: ")
    for produkt in wybrane_produkty:
        print(f"- {produkt['nazwa']}, Wartość: {produkt['wartosc']}zł, Waga: {produkt['waga']}kg")
    print(f"\nŁączna wartość: {current_wartosc}zł")
    print(f"\nŁączna waga: {current_waga}kg")
"""

# Knapsack - kradzież z plecakiem
"""
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])

    return dp[n][capacity]


# Test
weights = [1, 3, 4, 5]
values = [1, 4, 5, 7]
capacity = 7
print(knapsack(weights, values, capacity))  # Wynik: 9
"""

# ----------------------------------------------------------------------------------------------------------------------#
"""
Day: 5/10
Brakes_Count:
Goal_Completion: /31
Start_Data: 30/03/2025
End_Data: 10/04/2025
"""

# Bubble Sort
"""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # Flaga informująca, czy dokonano zamiany
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                # Zamiana elementów
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        # Jeśli w danym przebiegu nie dokonano zamiany, lista jest już posortowana
        if not swapped:
            break
        print(arr)
    return arr
lista = [64, 34, 25, 12, 22, 11, 90]
posortowana_lista = bubble_sort(lista)
print("Posortowana lista:", posortowana_lista)
"""

# silnia
"""
def s( x: int ) -> int:
    if x == 0 or x == 1: return 1
    else:
        temp = 1
        for i in range(2, x + 1):
            temp *= i
            print(temp)
        return temp
print(s(1))
"""
"""
def s_r(x: int) -> int:
    if x == 0 or x == 1: return 1
    else: return x * s_r(x - 1)
print(s_r(5))
"""
"""
import heapq

def dijkstra(graph, start):
    queue = [(0, start)]
    distances = {start: 0}
    while queue:
        (dist, node) = heapq.heappop(queue)
        for neighbor, weight in graph[node]:
            distance = dist + weight
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances
### ????????
"""

# NWD
"""
def najwiekszy_wspolny_dzielnik(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a
print(najwiekszy_wspolny_dzielnik(18, 81))
"""

# odwraca ciąg znaków
"""
def o_c_z(strs: str) -> str:
    return strs[::-1]
print(o_c_z("Jaka to Melodia!?!"))
"""

# licz ciąg znaków
"""
def count_string(strs: str, char: str) -> int:
    return strs.count(char)
print(count_string("Jaka to Melodia!?!", "a"))
"""

# Przykład DFS:
"""
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
"""

# slow w tekscie
"""
def slow_w_tekscie(tekst: str) -> int:
    return tekst.count(" ") + 1
print(slow_w_tekscie("Jaka to Melodia i coś jeszcze!?!"))
"""
"""
def count_words(text):
    return len(text.split())
"""
"""
a = [2,1,43,56,234,12]
a1 = map(lambda x: x * 2, a)
a2 = reduce(lambda n,y: n+y,a1)
print(a2)
"""
"""
def bs(li: list,x: int) -> int:
    l = 0
    h = len(li) - 1
    while l <= h:
        mid = (l+h)//2
        if li[mid] == x: return mid
        elif li[mid] < x: l = mid + 1
        else: h = mid - 1
print(bs([1,2,3,4,5,6,7,8,9,10],3))
"""
"""
def qs(arr: list) -> list:
    arr = list(set(arr)) # Jeżeli nie chcemy duplikatów
    if len(arr) < 2: return arr
    else: return qs([i for i in arr[1:] if i <= arr[0]]) + [arr[0]] + qs([i for i in arr[1:] if i > arr[0]])
print(qs([1,12212,31,231,312,3,125234,234,233,12312,3,53442314,23,233,123,13,123,123,12,234,12312,312,312,312,23423,323,3123,12]))
"""

# ----------------------------------------------------------------------------------------------------------------------#
"""
Day: 6/10
Brakes_Count:
Goal_Completion: /31
Start_Data: 30/03/2025
End_Data: 10/04/2025
"""

# --__--#


"""
def count_word(text, word):
    return text.lower().split().count(word.lower())
print(count_word("Jaka to Melodia i coś jeszcze!?!", "to"))
"""

# Algorytmy i Struktury Danych
# - Przygotowanie do Zawodów III Stopnia Olimpiady Innowacji Technicznych 2024/25

# Quick Sort (QS) - klasa O(n log n)
"""
def quick_sort(array: list) -> list:
    # array = list(set(array)) # Jeżeli chce się nie mieć powtórek
    if len(array) < 2: return array # 0 lub 1 elementowe tablice są już "posortowane", nie ma sensu się nimi zajmować...
    # Dodatkowo jest to przypadek bazowy, dla funkcji rekurencyjnej

    return quick_sort([element for element in array[1:] if element <= array[0]]) + [array[0]] + quick_sort([element for element in array[1:] if element > array[0]])
    # Zwracana jest lista przez konkatenację mniejszych i większych elementów od pierwszego elementu z brzegu wejściowej tablicy...
    # z pierwszym elementem listy między nimi
print(quick_sort([12,23,5,2,1,567,2,45,322,4,1,5,341,6,14,7,24,24,2,1234,45,34,234,124]))
"""

# Bubble Sort (BS) - klasa O( n^2 )
"""
def bubble_sort(array: list) -> list:
    if len(array) < 2: return array
    for i in range(len(array)):
        swapped = False
        for j in range(len(array) - i - 1):
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
                swapped = True
        if not swapped:
            break
    return array
print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))
"""

# Insert Sort (IS) - klasa O( n^2 )
"""
def insert_sort(array:list) -> list:
    if len(array) < 2: return array
    for i in range(len(array)):
        j = i
        temp = array[j]
        while j > 0 and array[j-1] > temp:
            array[j] = array[j-1]
            j -= 1
        array[j] = temp
    return array
print(insert_sort([64, 34, 25, 12, 22, 11, 90]))
"""

# Merge Sort (MS) - klasa O( n log n )
""" Sensu nie ma się na razie tego uczyć, za duże to jest
def merge_sort(left: int, right: int) -> list:
    if len(array) < 2: return array
    elif left < right:
        mid = (left+right) // 2
        merge_sort(left, mid)
        merge_sort(mid+1, right)
        scalaj(left, mid, right)
"""

# ----------------------------------------------------------------------------------------------------------------------#
"""
Day: 7/10
Brakes_Count:
Goal_Completion: /31
    - [X] Sort: QS, BS, IS,
    - [X] Search: BS
    - [X] Graph: BFS, DFS, Dij, Kru
    - [-] Geom.: Comex hull
    - [X] Common: GCD, Fi,!
    - [ ] Math: KMP, Rab-Karp, Mill-Rab
Start_Data: 30/03/2025
End_Data: 10/04/2025
"""

# --_----------________---__________-----------_--#

##-SORT-##

# QS - O( n log n )
"""
def qs(arr: list) -> list:
    if len(arr) < 2: return arr
    return qs([element for element in arr[1:] if element <= arr[0]]) + [arr[0]] + qs(
        [element for element in arr[1:] if element > arr[0]])
print(qs([3, 1, 2, 5, 9, 8, 6, 7]))
"""

# BS - O( n**2 )
"""
def bs(arr: list) -> list:
    if len(arr) < 2: return arr
    for i in range(len(arr)):
        swapped = False
        for j in range(0, len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
print(bs([3, 1, 2, 5, 9, 8, 6, 7]))
"""
"""
def bs(arr: list) -> list:
    if len(arr) < 2: return arr
    for i in range(len(arr)):
        swapped = False
        for j in range(0, len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
print(bs([3, 1, 2, 5, 9, 8, 6, 7]))
"""

# IS - O( n**2 )
"""
def i_s(arr: list) -> list:
    if len(arr) < 2: return arr
    for i in range(len(arr)):
        j = i
        key = arr[j]
        while j > 0 and arr[j - 1] > key:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = key
    return arr
print(i_s([64, 34, 25, 12, 22, 11, 90]))
"""
"""
def i_s(arr: list) -> list:
    if len(arr) < 2: return arr
    for i in range(len(arr)):
        j = i
        key = arr[j]
        while j > 0 and arr[j - 1] > key:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = key
    return arr
print(i_s([64, 34, 25, 12, 22, 11, 90]))
"""

##-SEARCH-##

# BS - O( log n )
"""
def bs(arr: list, element: int) -> int:
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == element:
            return mid
        elif arr[mid] < element:
            low = mid + 1
        else:
            high = mid - 1
print(bs([11, 12, 22, 25, 34, 64, 90], 34))
"""
# LS - śmieszne:
# ...w przypadku konwersji generatora na listę (np. za pomocą list(a)), złożoność przestrzenna będzie O(n),
# ponieważ przechowuje wszystkie wyniki w pamięci.
# Jednak sama funkcja ls jako generator ma złożoność przestrzenną O(1).
"""
def ls(arr: list, element: int):
    return (i for i in range(len(arr)) if arr[i] == element)
a = ls([11, 12, 22, 25, 34, 64, 90], 34)
print(*list(a))
"""

##-GRAPH-##

# Roy-Warshall/Floyd-Warshall - O( n**3 ) MAGIAAA
"""
graph = [
    [0, 3, float('inf'), float('inf')],
    [2, 0, float('inf'), 1],
    [float('inf'), 7, 0, 2],
    [6, float('inf'), 3, 0]
]
def roy_warshall(graph):
    n = len(graph)
    # Tworzymy kopię grafu, żeby nie modyfikować oryginalnej macierzy
    dist = [row[:] for row in graph]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
print(roy_warshall(graph))
"""

# BFS - O( V + E ) V = liczba wierzchołków, E = liczba krawędzi
# - zazwyczaj iteracyjnie, przez prostotę
"""
graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 4],
    3: [1],
    4: [1, 2]
}
from collections import deque
def bfs(graph, start):
    # Kolejka do przechowywania wierzchołków do odwiedzenia
    queue = deque([start])

    # Zbiór odwiedzonych wierzchołków
    visited = set([start])

    # Wynik - lista odwiedzonych wierzchołków
    result = []

    while queue:
        # Pobieramy wierzchołek z początku kolejki
        node = queue.popleft()
        result.append(node)

        # Przechodzimy przez sąsiadów wierzchołka
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result
result = bfs(graph, 0)
print(result)
"""

# DFS -  O(V + E) - zazwyczaj rekurencyjnie, przez naturę algorytmu

"""
graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 4],
    3: [1],
    4: [1, 2]
}


def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()  # Zbiór odwiedzonych wierzchołków

    # Oznaczamy wierzchołek jako odwiedzony
    visited.add(start)

    # Wydrukujemy lub przechowamy odwiedzony wierzchołek
    print(start, end=' ')

    # Rekurencyjnie odwiedzamy sąsiadów
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

    return visited
visited = dfs(graph, 0)

print("\nOdwiedzone wierzchołki:", visited)
"""

# Dijkstra - O( (V + E) log V )
"""
# Reprezentacja grafu jako słownik: każdy wierzchołek mapowany jest na listę (sąsiad, waga)
graph = {
    'A': [('B', 5), ('C', 1)],
    'B': [('A', 5), ('C', 2), ('D', 1)],
    'C': [('A', 1), ('B', 2), ('D', 4), ('E', 8)],
    'D': [('B', 1), ('C', 4), ('E', 3), ('F', 6)],
    'E': [('C', 8), ('D', 3)],
    'F': [('D', 6)]
}

import heapq

def dijkstra(graph, start):

    # Inicjalizacja odległości: początkowo nieskończoność dla wszystkich wierzchołków, oprócz startowego
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0

    # Kolejka priorytetowa: przechowuje krotki (aktualna_odległość, wierzchołek)
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        # Jeśli znaleziono już lepszą ścieżkę, pomijamy bieżący wierzchołek
        if current_distance > distances[current_vertex]:
            continue
        # Sprawdzamy sąsiadów bieżącego wierzchołka
        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight
            # Jeśli znaleziono krótszą ścieżkę do sąsiada, aktualizujemy odległość i dodajemy do kolejki
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances
start_vertex = 'A'
shortest_paths = dijkstra(graph, start_vertex)
print("Najkrótsze ścieżki od", start_vertex, ":", shortest_paths)
"""
"""
def dijkstra_simple(graph, start):
    # Inicjalizacja odległości: wszystkie wierzchołki mają początkowo nieskończoność, oprócz startowego
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0

    # Zbiór odwiedzonych wierzchołków
    visited = set()

    while len(visited) < len(graph):
        # Wybieramy nieodwiedzony wierzchołek o najmniejszej odległości
        current = None
        current_distance = float('inf')
        for vertex in graph:
            if vertex not in visited and distances[vertex] < current_distance:
                current_distance = distances[vertex]
                current = vertex

        # Jeśli nie znaleziono żadnego wierzchołka, kończymy działanie (pozostałe są nieosiągalne)
        if current is None:
            break

        # Oznaczamy bieżący wierzchołek jako odwiedzony
        visited.add(current)

        # Aktualizujemy odległości dla sąsiadów bieżącego wierzchołka
        for neighbor, weight in graph[current]:
            if neighbor in visited:
                continue
            new_distance = distances[current] + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance

    return distances
# Przykład użycia:
graph = {
    'A': [('B', 5), ('C', 1)],
    'B': [('A', 5), ('C', 2), ('D', 1)],
    'C': [('A', 1), ('B', 2), ('D', 4), ('E', 8)],
    'D': [('B', 1), ('C', 4), ('E', 3), ('F', 6)],
    'E': [('C', 8), ('D', 3)],
    'F': [('D', 6)]
}
start_vertex = 'A'
shortest_paths = dijkstra_simple(graph, start_vertex)
print("Najkrótsze ścieżki od", start_vertex, ":", shortest_paths)
"""

# Jeszcze prosciej Dijkstra
"""
def dijkstra(graph, start):
    # Inicjalizacja odległości: dla każdego wierzchołka nieskończoność, dla startu 0
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    used = set()
    while True:
        # Wybieramy nieprzetworzony wierzchołek o najmniejszej odległości
        u = None
        for v in graph:
            if v not in used and (u is None or dist[v] < dist[u]):
                u = v
        if u is None: break
        used.add(u)
        for v, weight in graph[u]:
            dist[v] = min(dist[v], dist[u] + weight)
    return dist
# Przykładowy graf: klucze to wierzchołki, wartości to listy par (sąsiad, waga)
graph = {
    'A': [('B', 5), ('C', 1)],
    'B': [('A', 5), ('C', 2), ('D', 1)],
    'C': [('A', 1), ('B', 2), ('D', 4), ('E', 8)],
    'D': [('B', 1), ('C', 4), ('E', 3), ('F', 6)],
    'E': [('C', 8), ('D', 3)],
    'F': [('D', 6)]
}
print(dijkstra(graph, 'A'))
"""


# Bellman-Ford - O( |V|*|E| )
"""
def bellman_ford(graph, start):
    # Inicjalizacja: wszystkie wierzchołki mają odległość nieskończoność, oprócz wierzchołka startowego
    dist = {vertex: float('inf') for vertex in graph}
    dist[start] = 0
    # Relaksacja wszystkich krawędzi (|V| - 1) razy
    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
    # Sprawdzenie cykli o ujemnej wadze
    for u in graph:
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                raise ValueError("Graf zawiera cykl o ujemnej wadze")
    return dist
    # Przykładowy graf:
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('C', 3), ('D', 2), ('E', 3)],
    'C': [('B', 1), ('D', 4), ('E', 5)],
    'D': [('E', -5)],
    'E': []
}
# Uruchomienie algorytmu Bellmana-Forda
start_vertex = 'A'
shortest_paths = bellman_ford(graph, start_vertex)
print("Najkrótsze ścieżki od", start_vertex, ":", shortest_paths)
"""

# Kruskal
"""
def kruskal(edges, vertices):
    # Funkcja realizująca algorytm Kruskala do wyznaczania minimalnego drzewa rozpinającego.
    # 
    # :param edges: Lista krawędzi, gdzie każda krawędź jest reprezentowana jako krotka (u, v, waga).
    # :param vertices: Lista lub zbiór wierzchołków grafu.
    # :return: Lista krawędzi należących do minimalnego drzewa rozpinającego.
    # Sortujemy krawędzie rosnąco według wagi
    edges.sort(key=lambda x: x[2])

    # Inicjalizacja struktury Union-Find
    parent = {v: v for v in vertices}
    rank = {v: 0 for v in vertices}

    def find(v):
        # Znajduje reprezentanta zbioru wierzchołka v z kompresją ścieżki.
        if parent[v] != v:
            parent[v] = find(parent[v])
        return parent[v]

    def union(u, v):
        # Łączy dwa zbiory reprezentowane przez u i v, stosując unii według rangi.
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            elif rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    mst = []  # Lista krawędzi w MST
    for u, v, w in edges:
        # Jeśli wierzchołki u i v należą do różnych zbiorów, dodajemy krawędź do MST
        if find(u) != find(v):
            union(u, v)
            mst.append((u, v, w))

    return mst


# Przykładowe użycie:
edges = [
    ('A', 'B', 1),
    ('A', 'C', 3),
    ('B', 'C', 1),
    ('B', 'D', 4),
    ('C', 'D', 2),
    ('C', 'E', 5),
    ('D', 'E', 1)
]
vertices = ['A', 'B', 'C', 'D', 'E']

mst = kruskal(edges, vertices)
print("Minimalne drzewo rozpinające:", mst)
"""

import heapq


# Prim
"""
def prim(graph, start):
    # Implementacja algorytmu Prima do wyznaczania minimalnego drzewa rozpinającego (MST).
    # 
    # :param graph: Słownik reprezentujący graf, gdzie klucze to wierzchołki, a wartości to listy krotek (sąsiad, waga).
    # :param start: Wierzchołek startowy.
    # :return: Lista krawędzi w MST w postaci (u, v, waga).
    mst = []  # Lista krawędzi w MST
    visited = {start}  # Zbiór odwiedzonych wierzchołków
    # Inicjujemy kolejkę priorytetową krawędzi wychodzących ze startu
    edges = [(weight, start, v) for v, weight in graph[start]]
    heapq.heapify(edges)

    while edges:
        weight, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, weight))
            # Dodajemy wszystkie krawędzie wychodzące z wierzchołka v do kolejki,
            # jeśli prowadzą do nieodwiedzonych wierzchołków.
            for to, w in graph[v]:
                if to not in visited:
                    heapq.heappush(edges, (w, v, to))

    return mst


# Przykładowe użycie:
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('A', 1), ('C', 1), ('D', 4)],
    'C': [('A', 3), ('B', 1), ('D', 1)],
    'D': [('B', 4), ('C', 1)]
}

mst = prim(graph, 'A')
print("Krawędzie MST:", mst)
"""

# A*
"""
import heapq


def a_star(graph, start, goal, h):
    # Znajduje najkrótszą ścieżkę od wierzchołka 'start' do 'goal' w grafie przy użyciu algorytmu A*.
    # 
    # :param graph: Słownik reprezentujący graf, np. {'A': [('B', 1), ('C', 4)], ...}
    # :param start: Wierzchołek początkowy.
    # :param goal: Wierzchołek docelowy.
    # :param h: Funkcja heurystyczna, która przyjmuje wierzchołek i zwraca szacowany koszt dojścia do 'goal'.
    # :return: Lista wierzchołków tworzących najkrótszą ścieżkę lub None, jeśli ścieżka nie istnieje.
    open_set = []
    heapq.heappush(open_set, (h(start), start))

    came_from = {}  # Słownik do rekonstrukcji ścieżki
    g_score = {vertex: float('inf') for vertex in graph}
    g_score[start] = 0

    f_score = {vertex: float('inf') for vertex in graph}
    f_score[start] = h(start)

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            # Rekonstrukcja ścieżki
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor, cost in graph[current]:
            tentative_g = g_score[current] + cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None


# Przykładowe użycie:
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}


def heuristic(vertex):
    # Przykładowa heurystyka - dla uproszczenia zwracamy 0 (A* staje się wtedy algorytmem Dijkstry)
    # W praktyce należy użyć funkcji szacującej koszt dojścia do celu.
    return 0


path = a_star(graph, 'A', 'D', heuristic)
print("Najkrótsza ścieżka:", path)
"""

##-GEOMETRY-##

# Convex Hull - skippppppppp


##-COMMON-##

# GCD(NWD)
"""
def gcd(a:int,b:int)->int:
    while b:
        a, b = b, a % b
    return a
print(gcd(101234543248, 24))
"""

# ! / Factorial / Silnia
"""
def fac(x: int)->int:
    if x == 0 or x == 1: return 1
    else:
        a = 1
        for i in range(2, x+1):
            a *= i
        return a
print(fac(5))
"""
"""
def fact(n:int)->int:
    if n==0 or n==1: return 1
    else:
        temp = 1
        for i in range(2, n+1):
            temp *= i
        return temp
print(fact(5))
"""
"""
a = lambda x: 1 if x == 0 or x == 1 else x * a(x -1)
print(silnia(5))
"""

# Fi
"""
def fi(n:int)->int:
    if n == 0 or n == 1: return 1
    else:
        a,b = 0, 1
        for i in range(2, n + 1):
            # print(a)
            a, b = b, a + b
            print(a)
        return b
print(fi(6))
"""
"""
def fif(x:int)->int:
    if x == 0 or x == 1: return 1
    else:
        a,b = 0, 1
        for i in range(2, n + 1):
            a, b = b, a + b
        return b
    """


##-MATH-##

# KMP -
"""
def kmp_search(text, pattern):
    m, n = len(pattern), len(text)
    i,j = 0,0  # Indeks w tekście Indeks w wzorcu
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:  # Wzorzec został znaleziony
                print(f'Pattern found at index {i - j}')
                j = 0  # Restartujemy wyszukiwanie wzorca
        else:
            if j != 0: j = 0  # Restartujemy wzorzec, jeśli występuje niezgodność
                # Zamiast przesuwać tekst, wracamy do kolejnego możliwego dopasowania
                # przez przesunięcie wzorca na początek
            else: i += 1  # Przesuwamy wskaźnik w tekście, jeśli wystąpiła niezgodność
# Przykładowe użycie:
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
kmp_search(text, pattern)
"""


# ----------------------------------------------------------------------------------------------------------------------#
"""
Day: 8/10
Brakes_Count:
Goal_Completion: /31
    - [ ] Sort: QS, BS, IS,
    - [ ] Search: BS
    - [ ] Graph: BFS, DFS, Dij, Kru
    - [ ] Common: GCD, Fi,!
    - [ ] Text: KMP, Rab-Karp, Mill-Rab
Start_Data: 30/03/2025
End_Data: 10/04/2025
"""

# --_----------________---__________-----------_--#

##-SORT-##

# QS - O( n log n )
"""
def qs(arr:list)->list:
    arr = list(set(arr))
    if len(arr) < 2: return arr
    return qs([el for el in arr[1:] if el <= arr[0]]) + [arr[0]] + qs([el for el in arr[1:] if el > arr[0]])
print(qs("ABABDABACDABABCABAB"))
"""

# BS - O( n**2 )
"""
def bs(arr:list)->list:
    if len(arr) < 2: return arr
    for i in range(len(arr)):
        swap = False
        for j in range(0, len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swap = True
        if not swap:
            break
    return arr
print(bs([3, 1, 2, 5, 9, 8, 6, 7]))
"""
"""
def bs2(lst: list) -> list:
    if len(lst) < 2: return lst
    for i in range(len(lst)):
        swa = False
        for j in range(0, len(lst) - i - 1):
            if lst[j] > lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
                swa = True
        if not swa: break
    return lst
print(bs2([3, 1, 2, 5, 9, 8, 6, 7]))
"""

# IS - O( n**2 )
"""
def IS(arr: list) -> list:
    if len(arr) < 2: return arr
    for i in range(len(arr)):
        j = i
        key = arr[j]
        while j > 0 and arr[j - 1] > key:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = key
    return arr
print(IS([3, 1, 2, 5, 9, 8, 6, 7]))
"""

##-SEARCH-##
"""
def bs(lst: list, ele: int) -> int:
    low = 0
    high = len(lst) - 1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] == ele: return mid
        elif lst[mid] < ele: low = mid + 1
        else: high = mid - 1
print(bs([1,2,3,4,5,6,7,8,9,10],3))
"""

##-COMMON-##

# Bin
"""
def int_bin(x: int) -> None:
    if x == 0:
        print(0)
        return
    result = ""
    while x:
        result = str(x % 2) + result
        x //= 2
    print(result)

int_bin(100)
"""
"""
def i_b(x: int) -> None:
    if x == 0:
        print(0)
        return
    res = []
    while x:
        res.append(x & 1)
        x //= 2
    res.reverse()
    print(*res, sep="")
i_b(100)
"""
"""
def ib2(x):
    if x == 0:
        print(x)
        return
    res = ""
    while x:
        res = str(x % 2) + res
        x //= 2
    print(res)
ib2(100)
"""
"""
def sil(x:int)->int:
    if x == 0 or x == 1: return 1
    temp = 1
    for i in range(2,x + 1):
        temp *= i
        print(f"Liczba: {temp} Mnożone przez: {i}")
    return temp
sil(6)
"""





















