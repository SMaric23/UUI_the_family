# Analiza Sentimenta Recenzija – The Family

Implementacija modela za automatsku analizu sentimenta klijenata ugostiteljskog objekta "The Family" koristeći tehnike nadziranog učenja (SVM + TF-IDF).

## 🎯 Cilj

Razviti sustav koji automatski klasificira recenzije restorana kao:
- **POZITIVAN** (ocjena 4–5 zvjezdica, NPS 9-10)
- **NEGATIVAN** (ocjena 1–3 zvjezdice, NPS 0-6)

Omogućiti brzu analizu recenzija i automatizaciju obrade povratnih informacija.

---

## 🚀 Brzi Start

### Instalacija

```bash
pip install -r requirements.txt
```

### Pokretanje

```bash
python main.py
```

**Napomena:** Trebala bi datoteka `the_family_review.xlsx` u istoj mapi!

---

## 📊 Rezultati

Performanse modela na testnom skupu:

```
Točnost (accuracy):   89.66%
Preciznost:           0.9085
Odziv (recall):       0.8966
F1-mjera:             0.8825
```

**Matrica zabune:**
```
                        Predviđeno: NEG  Predviđeno: POZ
Stvarno: NEGATIVAN              3             3
Stvarno: POZITIVAN              0            23
```

---

## 📁 Datoteke

- **`main.py`** – Kompletan script (klasa + funkcije + main)
- **`requirements.txt`** – Python zavisnosti
- **`README.md`** – Dokumentacija
- **`dokumentacija.tex`** – LaTeX dokumentacija (14+ stranica)
- **`LICENSE`** – GPL v3 licenca
- **`the_family_review.xlsx`** – Dataset (150+ recenzija)

---

## 🛠️ Tehnologije

- **Python 3.8+**
- **scikit-learn** – SVM + TF-IDF model
- **pandas** – Obrada podataka
- **numpy** – Numeričke operacije

---

## 🔍 Kako Radi

### 1. Obrada Podataka
```python
# Mapiranje ocjena (1-5) na NPS (0-10)
1 ⟹ 2 (NEGATIVAN)
2 ⟹ 4 (NEGATIVAN)
3 ⟹ 6 (NEGATIVAN)
4 ⟹ 8 (NEUTRALAN)
5 ⟹ 10 (POZITIVAN)
```

### 2. Model
```
Tekst Recenzije
    ↓
TF-IDF Vektorizacija (5000 obilježja)
    ↓
Linear SVM (C=1.0, max_iter=2000)
    ↓
POZITIVAN / NEGATIVAN
```

### 3. Testiranje
```python
klassifikator = SentimentKlasifikator()
klassifikator.treniraj(X_train, y_train)

sentiment = klassifikator.predikat("Odličan restoran!")
# Output: 'POZITIVAN'
```

---

## 📚 Sastavnice

### `SentimentKlasifikator` klasa

**Metode:**
- `treniraj(X, y)` – Trenira model
- `predikat(tekst)` – Predikat za jednu recenziju
- `predikat_batch(tekstovi)` – Predikat za više recenzija
- `ispis_metrika()` – Ispis detaljnih metrika
- `matrica_zabune()` – Matrica zabune kao DataFrame
- `spremi_model(putanja)` – Spremi model na disk

### `pripremi_podatke(excel_datoteka)`

Učitava Excel datoteku i transformira je:
- Uklanja prazne redove
- Mapira ocjene na NPS ljestvicu
- Klasificira sentimente

---

## ⚙️ Parametri Modela

### TF-IDF Vectorizer
```python
max_features=5000      # Max riječnih obilježja
ngram_range=(1, 2)    # Unigrams + bigrams
min_df=2              # Min dokumenata sa riječju
max_df=0.9            # Max dokumenata sa riječju (90%)
```

### Linear SVM
```python
C=1.0                 # Parametar regularizacije
max_iter=2000         # Max iteracija
random_state=42       # Za reproducibilnost
```

---

## 📝 Licenca

**GPL v3** – Slobodno koristiš, ali dijeli izmjene!

---

## 👨‍💻 Autor

**Projekt za Kolegij:** Učenje Indukcijom (Artificial Intelligence)  
**Institucija:** FOI – Fakultet Organizacije i Informatike  
**Datum:** 2026

---

## ❓ FAQ

**P: Mogu li koristiti drugačiji dataset?**  
O: Da! Trebam Excel datoteku sa stupcima `review_text`, `review_rating`, `review_datetime_utc`.

**P: Kako poboljšam točnost?**  
O: 
- Dodaj više primjera za trening
- Primijeni lemmatizaciju (hrvatskog jezika)
- Koristi drugačite SVM parametre (C, kernel)
- Isprobaj drugih modela (Naive Bayes, Random Forest)

**P: Mogu li koristiti model bez retraininga?**  
O: Da! Isti model je učitan sa `sentiment_model.pkl` – vidi dokumentaciju.

---

## 🔗 Kontakt

GitHub: https://github.com/SMaric23/UUI_the_family
