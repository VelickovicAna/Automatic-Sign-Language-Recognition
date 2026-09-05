# Automatic-Sign-Language-Recognition

Cilj ovo projekta je automatsko prepoznavanje slova američkog znakovnog jezika (American Sign Language - ASL) sa slika ruku, uz poređenje dva različita pristupa:

1. **Bazni CNN model** - direktna klasifikacija slike u jedno od 24 slova.
2. **Metric learning (Triplet Loss) + KNN** - konvolutivna mreža uči da slike preslika u vektorski prostor (embedding) u kome su slike istog slova jedna blizu druge, a KNN klasifikator nad tim embedinzima vrši predikciju.

## Skup podataka

- Izvor: [Sign Language MNIST (Kaggle)](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- 27 455 trening i 7 172 test primera, slike 28x28 piksela u sivim tonovima, zapisane kao redovi piksela u CSV formatu.
- 24 klase (slova A-Z), bez slova **J** i **Z** jer njihovo predstavljanje u znakovnom jeziku zahteva pokret, a skup sadrži samo statične slike.
- Trening skup je približno balansiran (950-1300 primera po klasi); test skup je neravnomerniji (140-500 po klasi).

Skup podataka **nije uključen u repozitorijum** (prevelik i licenciran preko Kaggle-a). Za pokretanje projekta potrebno je:

1. Preuzeti `sign_mnist_train.csv` i `sign_mnist_test.csv` sa Kaggle stranice iznad.
2. Sačuvati ih u `data/raw/`.

## Okruženje i pokretanje

Python 3.10+ se preporučuje za pokretanje projekta.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Notebook-ovi se pokreću redom (svaki naredni koristi izlaz prethodnog, sačuvan u `data/processed/` ili `results/`):

1. `01_eda.ipynb`
2. `02_priprema_podataka.ipynb`
3. `03_cnn.ipynb`
4. `04_triplet_embedding.ipynb`
5. `05_KNN.ipynb`
6. `06_poredjenje_modela.ipynb`

## Rezultati

Oba modela postižu praktično identičnu i veoma visoku tačnost na test skupu (razlika je na trećoj/četvrtoj decimali); 
detaljne metrike, matrice konfuzije i analiza grešaka po slovima nalaze se u `results/` kao i u `notebooks/06_poredjenje_modela.ipynb`.

## Literatura i reference

- Sign Language MNIST dataset: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
- Schroff, F., Kalenichenko, D., Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering* (triplet loss). https://arxiv.org/abs/1503.03832
- pytorch-metric-learning dokumentacija: https://kevinmusgrave.github.io/pytorch-metric-learning/
- scikit-learn dokumentacija (KNN, metrike): https://scikit-learn.org/stable/
- Mladen Nikolić, Anđelka Zečević, Mašinsko učenje: https://ml.matf.bg.ac.rs/readings/ml.pdf

## Tim

- Ana Veličković 1128/2025,
- Luna Rančić 1027/2025
