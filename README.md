# Breast Cancer Wisconsin
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

## Citation
```
This breast cancer databases was obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg. If you publish results when using this database, then please include this information in your acknowledgements. Also, please cite one or more of:
1. O. L. Mangasarian and W. H. Wolberg: "Cancer diagnosis via linear programming", SIAM News, Volume 23, Number 5, September 1990, pp 1 & 18.
2. William H. Wolberg and O.L. Mangasarian: "Multisurface method of pattern separation for medical diagnosis applied to breast cytology", Proceedings of the National Academy of Sciences, U.S.A., Volume 87, December 1990, pp 9193-9196.
3. O. L. Mangasarian, R. Setiono, and W.H. Wolberg: "Pattern recognition via linear programming: Theory and application to medical diagnosis", in: "Large-scale numerical optimization", Thomas F. Coleman and Yuying Li, editors, SIAM Publications, Philadelphia 1990, pp 22-30.
4. K. P. Bennett & O. L. Mangasarian: "Robust linear programming discrimination of two linearly inseparable sets", Optimization Methods and Software 1, 1992, 23-34 (Gordon & Breach Science Publishers).
```

Confusion matrix

BEST FEATURES
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])



(dla wybranej topologii lub kilku na wyższą ocenę)
-
Nie usunięcie kolumny „id” zmniejsza skuteczność klasyfikatorów do rzutu monetą / lub klasy 
Podmienienie brakujących danych na -999, 0, lub medianę wartości kolumny nie wpływa na skuteczność klasyfikatorów
- dla wybranej bazy danych testowane klasyfikatory osiągały zbliżoną skuteczność
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
