Breast Cancer Wisconsin
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

Eksperymenty muszą być w czytelny sposób opisane w
sprawozdaniu PDF, a projekt (kod) przechowujemy na githubie / bitbucket i
podajemy link w sprawozdaniu (można mnie dodać: gmadejsk @ github,
grzesiekm @ bitbucket).

W sprawozdaniu należy uwzględnić rozdział wstępny, w którym:
• Opiszesz, jakie dane zawiera wybrana baza danych i jakiej klasyfikacji
dokonujemy (która kolumna, jakie wartości).
• Dla każdej z kolumny podasz podstawowe informacje: min, max, średnia i
częstość występowania poszczególnych odpowiedzi (np. na wykresie
kołowym lub słupkowym). W przypadku kolumn numerycznych można
podzielić to na przedziały. Wskażesz też na procent brakujących danych.
• Wyjaśnisz jak baza danych została przygotowana do klasyfikacji. Czy jakieś
kolumny zostały zmodyfikowane? Usunięte? Czy wykryto jakieś błędne
dane?

Porównanie poznanych klasyfikatorów (wymagane na ocenę 3.0)
Główny rozdział sprawozdania to porównanie skuteczności klasyfikatorów na
bazie danych.
• Podziel bazę danych na zbiór testowy i treningowy. Ewaluacji wszystkich
klasyfikatorów dokonuj na jednym zbiorze testowym.
• Przetestuj klasyfikatory poznane na zajęciach
o Naive Bayes
o Drzewa decyzyjne
o k Najbliższych sąsiadów (dla wybranego k)
o Sieci neuronowe (dla wybranej topologii)
• Ewaluacja powinna zawierać dokładność klasyfikatora i macierz błędu.
• Wskaż najlepiej działający klasyfikator.
Rozszerzona klasyfikacja i inne techniki (wymagane na ocenę 4.0 i 5.0)
Tak jak w wymaganiach na ocenę 3, ale wymagania są rozszerzone. Im więcej
poniższych podpunktów uwzględnisz, tym wyższa będzie ocena.
• Dodaj kilka innych klasyfikatorów:
o k Najbliższych sąsiadów (dla kilku wybranych k)
o Sieci neuronowe (dla kilku wybranych topologii)
o Support Vector Machines
o Random Forest
o Metody typu Ensemble
o Inne?
Dla nowych klasyfikatorów zrób krótki wstęp teoretyczny. Wyjaśnij na jakiej
zasadzie działają.
• Rozszerzona ewaluacja klasyfikatorów. Jak inaczej można oceniać
klasyfikatory? Które miary będą miały sens w Twoich badaniach? Na
początek można rzucić okiem na:
https://en.wikipedia.org/wiki/Sensitivity_and_specificity i poszukać innych
źródeł rozwijających temat.
• Szukanie reguł asocjacyjnych. Czy ma sens? Jakich najlepiej szukać? Podaj
te dla nas szczególnie interesujące.
• Porównanie skuteczności klasyfikatorów na jakichś wykresach. Słupkowy?
ROC?
• Czy w naszej bazie danych jakieś rodzaje błędów są ważniejsze /
poważniejsze niż inne? Dlaczego?

