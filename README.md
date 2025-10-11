# SOM izveštaj

## Uvod

Ovaj dokument predstavlja izveštaj i ujedno dokumentaciju za projekat iz predmeta **Sistemi odlučivanja u medicini**. Dodatno, ovaj dokument služi i kao kompletno uputstvo za pokretanje i korišćenje primenjenog use-case-a klasifikacije.

## Preduslovi

Da bi se projekat pokrenuo nephodno je imati:
- Python interpreter
- Python IDE
- Git

## Instalacija projekta

Klonirajte projekat u vašem root folderu:

```
https://github.com/fil-SK/MedClassifier.git
```

U okviru istog root foldera, potrebno je klonirati i [MedMNIST](https://github.com/MedMNIST/MedMNIST) repo:

```
https://github.com/MedMNIST/MedMNIST.git
```

## Preduslovi za pokretanje

Neophodno je instalirati sve potrebne pakete, date kroz `requirements.txt` dokument. To je moguće uraditi kroz:

```
pip install requirements.txt
```

## Projekat: zahtevi, struktura, pokretanje

### Zahtevi

Projekat treba da predstavi use-case primene neuralnih mreža u oblasti medicine. Odabrano je da se radi zadatak **klasifikacije**, na dobro poznatom medicinskom datasetu (MedMNIST), koristeći PyTorch framework. Ovaj projekat može se primeniti na proizvoljan klasifikacioni model, ali je primarni use-case na ResNet101 modelu.

### Struktura

Finalni izgled fajl sistema projekta izgleda ovako:

```
root
|--- .venv
|--- README.md
|--- main.py
|--- support_scripts.py
|--- MedMNIST
|--- TissueMNIST_Dataset
        |--- test
            |--- tissuemnist.npz
        |--- train
            |--- tissuemnist.npz
        |--- val
            |--- tissuemnist.npz
```

Pojašnjenje:
- `main.py` - Glavna skripta koja se pokreće. Predstavlja početnu tačku projekta, iz nje se pozivaju sve potrebne funkcionalnosti.
- `support_scripts.py` - Python fajl koji sadrži sve pomoćne funkcije, radi čistijeg i preglednijeg koda.
- `MedMNIST` - Folder koji sadrži klonirani MedMNIST repo.
- `TissueMNIST_Dataset` - Folder koji sadrži slike korišćene za proces klasifikacije, grupisane u podfoldere koji predstavljaju podskupove glavnog dataseta. Sadržaj foldera čini `tissuemnist.npz`, što je NumPy kontejner.
- `tissuemnist.npz` - NumPy kontejner koji sadrži više `.npy` fajlova u arhivi. Svaki podfolder sadrži njemu relevantne slike.