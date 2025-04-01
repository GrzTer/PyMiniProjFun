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

quick_sort = lambda arr: arr if len(arr) < 2 else quick_sort([i for i in arr[1:] if i <= arr[0]]) + [arr[0]] + [i for i in arr[1:] if i > arr[0]]
print(f"Posortowana lista - quicksort: {quick_sort([3, 1, 2, 4, 0, 9, -10, -1, 6, 2, 13, 6, 8, 7])}")





