class Kalkulator:
    def __init__(self, wartosc=0):
        self.wartosc = wartosc

    def __repr__(self):
        return f'Kalkulator({self.wartosc})'

    def __add__(self, other):
        if isinstance(other, Kalkulator):
            return Kalkulator(self.wartosc + other.wartosc)
        return Kalkulator(self.wartosc + other)

    def __sub__(self, other):
        if isinstance(other, Kalkulator):
            return Kalkulator(self.wartosc - other.wartosc)
        return Kalkulator(self.wartosc - other)

    def __mul__(self, other):
        if isinstance(other, Kalkulator):
            return Kalkulator(self.wartosc * other.wartosc)
        return Kalkulator(self.wartosc * other)

    def __truediv__(self, other):
        if isinstance(other, Kalkulator):
            if other.wartosc != 0:
                return Kalkulator(self.wartosc / other.wartosc)
            else:
                raise ValueError("Nie można dzielić przez zero")
        if other != 0:
            return Kalkulator(self.wartosc / other)
        else:
            raise ValueError("Nie można dzielić przez zero")

    def __floordiv__(self, other):
        if isinstance(other, Kalkulator):
            if other.wartosc != 0:
                return Kalkulator(self.wartosc // other.wartosc)
            else:
                raise ValueError("Nie można dzielić przez zero")
        if other != 0:
            return Kalkulator(self.wartosc // other)
        else:
            raise ValueError("Nie można dzielić przez zero")

    def __mod__(self, other):
        if isinstance(other, Kalkulator):
            if other.wartosc != 0:
                return Kalkulator(self.wartosc % other.wartosc)
            else:
                raise ValueError("Nie można dzielić przez zero")
        if other != 0:
            return Kalkulator(self.wartosc % other)
        else:
            raise ValueError("Nie można dzielić przez zero")

    def __pow__(self, other):
        if isinstance(other, Kalkulator):
            return Kalkulator(self.wartosc ** other.wartosc)
        return Kalkulator(self.wartosc ** other)

    def __eq__(self, other):
        if isinstance(other, Kalkulator):
            return self.wartosc == other.wartosc
        return self.wartosc == other

