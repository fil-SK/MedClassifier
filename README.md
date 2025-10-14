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
pip install -r requirements.txt
```

## Projekat: zahtevi, struktura, pokretanje

### Zahtevi

Projekat treba da predstavi use-case primene neuralnih mreža u oblasti medicine. Odabrano je da se radi zadatak **klasifikacije**, na dobro poznatom medicinskom datasetu (MedMNIST), koristeći PyTorch framework. Ovaj projekat može se primeniti na proizvoljan klasifikacioni model, ali je primarni use-case na ResNet101 modelu.

Kod u projektu predstavlja kombinaciju javno dostupnog koda iz MedMNIST repoa, PyTorch frameworka i potrebnih funkcija koje obezbeđuju tražene funkcionalnosti.

### Spoljašnji alati

#### Posmatrani dataset

Posmatra se jedan podskup krovnog [MedMNIST](https://medmnist.com/) dataseta - TissueMNIST.

- **Modalitet podataka**: Bubrežni korteks pod mikroskopom (*Kidney Cortex Microscope*)
- **Ukupni broj uzoraka**: 236.386
- **Train / Validation / Test raspodela uzoraka**: 165.466 / 23.640 / 47.280
- **Klase**: Multi-klasna klasifikacija (8):

    - Collecting Duct, Connecting Tubule
    - Distal Convoluted Tubule
    - Glomerular endothelial cells
    - Interstitial endothelial cells
    - Leukocytes
    - Podocytes
    - Proximal Tubule Segments
    - Thick Ascending Limb

#### PyTorch framework

Korišćen je PyTorch kao okvirni framework za rad i njihov ugrađeni [ResNet101 model](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html#ResNet101_Weights), sa pretreniranim weight-ovima.

#### Torchmetrics

Za analizu performansi, korišćena je metrika preciznosti (accuracy), kroz biblioteku [Torchmetrics](https://lightning.ai/docs/torchmetrics/stable/), podržanu kroz korišćeni PyTorch framework.

#### Kalibracija hiperparametara

Kako bi se izbeglo ručno kalibrisanje hiperparametara, odlučeno je da se koristi neki od mogućih framework-ova namenjenih za tu svrhu. Odabrano je da se koristi [Optuna](https://optuna.org/).

### Struktura

Finalni izgled fajl sistema projekta izgleda ovako:

```
root
|--- .venv
|--- README.md
|--- main.py
|--- exported_models
        |--- model_20251012_173336.pt
|--- result_logs
        |--- log_21-03_12-10-2025.txt
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

## Tok programa

Sav kod koji se izvršava nalazi se u `main.py` fajlu, a koji interno poziva skripte i parametre iz `support_scripts.py` fajla. `main.py` ima delove koji su zakomentarisani, a koji mogu biti korisni pri pregledu:

- `print_dataset_info()`: Ispisuje informacije o korišćenom datasetu.
- `view_dataset_contents(train_dataset)`: Preko matplotlib-a prikazuje sliku i njoj odgovarajuću labelu, za proizvoljno uzetu sliku iz dataseta.
- `train_dataset.montage(length=10)`: Prikazuje 100 slika iz dataseta.

Izvršavanje `main.py` sastoji se iz (uključujući i zakomentarisane delove):

- Ispis informacija o korišćenom datasetu.
- Kreiranje foldera u kom će se naći dataset(ovi) - svaki skup (train, val, test) ima svoj folder. Preuzimanje datih skupova.
- Prikaz sadržaja dataseta.
- Enkapsulacija dataseta u dataloader-e za dalju upotrebu.
- Vizualizacija dataseta kroz 100 slika.
- Instanciranje modela i promenu finalnog FC sloja, tako da output-uje 8 klasa (toliko klasa postoji u korišćenom datasetu).
- Izvršavanje klasifikacije na originalnom, netreniranom modelu.
- Treniranje modela: po epohi, vrši se trening a potom validacija izvršenog treninga u toj epohi.
- Nakon treniranja u potpunosti, validacija takvog modela i njegovo čuvanje.

## Google Colab

Na lokalnom računaru, resursi su nedovoljni za efikasno izvršavanje (nedostatak GPU-a). Iz tog razloga, ovaj kod može se pokrenuti i kroz Google Colab.

Neophodno je upload-ovati sve sadržaje root foldera (nakon svih kloniranja projekta i potrebnih pomoćnih repoa, kao i preuzimanja dataseta) na Vaš Google Drive.

Nakon toga, napraviti Colab stranicu i na njoj mount-ovati uploadovan sadržaj:

```
from google.colab import drive
drive.mount('/content/gdrive')
```

Instalirati sve potrebne pakete:

```
!pip install -r /content/gdrive/MyDrive/som_proj/requirements.txt
```

Zatim, na putanji na kojoj se nalazi `main.py` fajl, pokrenuti ga:

```
!python /content/gdrive/MyDrive/som_proj/main.py
```

Dalje, izvršavanje će teći u potpunosti kao i na lokalnom računaru.

Da bi se omogućilo izvršavanje na GPU, neophodno je omogućiti to kroz Google Colab:

- Odabrati tab `Runtime`, pa odatle `Change runtime type`.
- Odabrati `T4 GPU`.
- Kliknuti na `Save`.

## Rezultati

### log 20.51. 14.10.2025.

- Korišćen Optuna framework za parametrizaciju, ali za Resnet18 model. Model je pokazao brže poboljšanje, ali sam ubrzo obustavio testiranje, jer želim da probam drugačiji pristup pri treniranju.
- Cilj: Trenirati od nule, bez pretreniranih težina (postoji mogućnost da ovde oni odmažu) i trenirati na značajno većem broju epoha (50).

### log 00.10. 14.10.2025.

- Korišćen Optuna framework za parametrizaciju hiperparametara. Usled nedostatka vremena model je pušten na samo 5 trial-a. Uspeo je da pronađe parametre koje su preciznost podigli na ~60%.
- Međutim, ovo je i dalje neočekivana performantnost. Postoji mogućnost da je ResNet101 previše složen model, što utiče na to da overfituje. U kasnijim istraživanjima isprobati manje dublje Resnet modele - 18 i 50.

### log 21.03 12.10.2025.

- Prvi test za klasifikaciju i optimizaciju modela. Rezultati zasad nedovoljno dobri. Inicijalno, model je pokazao preciznost od 10%. Treniranjem modela preciznost podignuta na oko 50%.
- Primenjeni hiperparametri:
  - Broj epoha: 3
  - Learning rate: 0.001
  - Batch size: 128
  - Model: ResNet101
  - Loss funkcija: Cross Entropy Loss
  - Optimizer: SGD
- Vreme izvršavanja: Približno 2h.