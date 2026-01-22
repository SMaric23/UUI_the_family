# main.py
# ============================================================
# SENTIMENT ANALIZA RECENZIJA - KOMPLETAN SCRIPT
# ============================================================

import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class SentimentKlasifikator:
    """Klasifikator sentimenta recenzija - TF-IDF + Linear SVM"""
    
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )),
            ('svm', LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
                verbose=0
            ))
        ])
        self.je_istreniran = False
        self.metrike = {}
        self.X_test = None
        self.y_test = None
        self.y_pred = None
    
    def treniraj(self, X, y, test_size=0.2, random_state=42):
        """Trenira model na podacima."""
        print("⏳ Treniranje modela...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"Broj primjera u trening skupu: {len(X_train)}")
        print(f"Broj primjera u testnom skupu: {len(X_test)}")
        
        self.model.fit(X_train, y_train)
        self.je_istreniran = True
        
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)
        
        self.metrike = {
            'accuracy': accuracy_score(y_test, self.y_pred),
            'precision': precision_score(y_test, self.y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, self.y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, self.y_pred, average='weighted', zero_division=0)
        }
        
        print("✅ Model je istreniran!")
        return self.metrike
    
    def predikat(self, tekst):
        """Predikat sentiment za jednu recenziju."""
        if not self.je_istreniran:
            raise ValueError("Model nije istreniran!")
        return self.model.predict([tekst])[0]
    
    def predikat_batch(self, tekstovi):
        """Predikat sentiment za više recenzija."""
        if not self.je_istreniran:
            raise ValueError("Model nije istreniran!")
        return self.model.predict(tekstovi)
    
    def ispis_metrika(self):
        """Ispiši metrike."""
        if not self.je_istreniran:
            print("❌ Model nije istreniran!")
            return
        
        print("\n📊 METRIKE NA TESTNOM SKUPU")
        print("="*50)
        print(f"Točnost (accuracy):  {self.metrike['accuracy']:.4f} ({self.metrike['accuracy']*100:.2f}%)")
        print(f"Preciznost:          {self.metrike['precision']:.4f}")
        print(f"Odziv (recall):      {self.metrike['recall']:.4f}")
        print(f"F1-mjera:            {self.metrike['f1']:.4f}")
        print("="*50)
        
        print("\n📋 Detaljan izvještaj:")
        print(classification_report(self.y_test, self.y_pred))
    
    def matrica_zabune(self):
        """Vrati matricu zabune."""
        if not self.je_istreniran or self.y_pred is None:
            return None
        
        cm = confusion_matrix(self.y_test, self.y_pred, labels=['NEGATIVAN', 'POZITIVAN'])
        return pd.DataFrame(
            cm,
            index=['Stvarno: NEGATIVAN', 'Stvarno: POZITIVAN'],
            columns=['Predviđeno: NEGATIVAN', 'Predviđeno: POZITIVAN']
        )
    
    def spremi_model(self, putanja='sentiment_model.pkl'):
        """Spremi model na disk."""
        if not self.je_istreniran:
            print("❌ Nema što spremati!")
            return
        with open(putanja, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model spremljen: {putanja}")


def pripremi_podatke(excel_datoteka):
    """Pripremi podatke iz Excel datoteke."""
    print("📂 Učitavanje podataka...")
    df = pd.read_excel(excel_datoteka)
    
    df_recenzije = df[['review_text', 'review_rating', 'review_datetime_utc']].copy()
    df_recenzije = df_recenzije.dropna(subset=['review_text', 'review_rating'])
    print(f"✅ Učitano {len(df_recenzije)} recenzija")
    
    df_recenzije = df_recenzije.reset_index(drop=True)
    df_recenzije['ID'] = df_recenzije.index + 1
    
    df_recenzije = df_recenzije.rename(columns={
        'review_text': 'Recenzija',
        'review_rating': 'Ocjena',
        'review_datetime_utc': 'Datum'
    })
    
    def ocjena_u_nps(ocjena):
        mapa = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10}
        return mapa.get(int(ocjena), 6)
    
    def nps_u_sentiment(nps_ocjena):
        if nps_ocjena <= 6:
            return 'NEGATIVAN'
        elif nps_ocjena <= 8:
            return 'NEUTRALAN'
        else:
            return 'POZITIVAN'
    
    df_recenzije['NPS_ocjena'] = df_recenzije['Ocjena'].apply(ocjena_u_nps)
    df_recenzije['Sentiment'] = df_recenzije['NPS_ocjena'].apply(nps_u_sentiment)
    
    print("\nDistribucija sentimenta:")
    print(df_recenzije['Sentiment'].value_counts())
    
    return df_recenzije


def main():
    """Glavna funkcija - pokreni sve."""
    print("="*60)
    print("🧠 ANALIZA SENTIMENTA RECENZIJA - THE FAMILY")
    print("="*60)
    
    # 1. Pripremi podatke
    df_recenzije = pripremi_podatke('the_family_review.xlsx')
    
    # 2. Filtriraj za model
    df_model = df_recenzije[df_recenzije['Sentiment'] != 'NEUTRALAN'].copy()
    print(f"\n📊 Skup za model: {len(df_model)} recenzija")
    print(f"   POZITIVAN: {(df_model['Sentiment'] == 'POZITIVAN').sum()}")
    print(f"   NEGATIVAN: {(df_model['Sentiment'] == 'NEGATIVAN').sum()}")
    
    # 3. Trenira model
    klasifikator = SentimentKlasifikator()
    X = df_model['Recenzija'].astype(str)
    y = df_model['Sentiment']
    
    metrike = klasifikator.treniraj(X, y, test_size=0.2, random_state=42)
    
    # 4. Ispis metrika
    klasifikator.ispis_metrika()
    
    # 5. Matrica zabune
    print("\n🎯 Matrica zabune:")
    print(klasifikator.matrica_zabune())
    
    # 6. Test predviđanja
    print("\n🔮 TESTIRANJE NA NOVIM RECENZIJAMA:")
    print("="*60)
    
    test_recenzije = [
        "Hrana je bila savršena, osoblje ljubazno, ambijent predivan!",
        "Loše iskustvo, hrana neukusna, osoblje neljubazno.",
        "Cijene previsoke za kvalitetu koja se nudi.",
        "Odličan restoran, sve preporuke! Vraćam se sigurno!"
    ]
    
    for tekst in test_recenzije:
        sentiment = klasifikator.predikat(tekst)
        print(f"Recenzija: \"{tekst[:70]}...\"")
        print(f"Sentiment: {sentiment}\n")
    
    # 7. Spremi model
    klasifikator.spremi_model('sentiment_model.pkl')
    
    print("="*60)
    print("✅ Analiza je završena!")
    print("="*60)


if __name__ == "__main__":
    main()
