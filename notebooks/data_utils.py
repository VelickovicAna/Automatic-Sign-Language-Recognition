from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# Konstante (identične kao kao u 02_priprema_podataka.ipynb)

# putanje
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_NPZ = PROCESSED_DIR / "sign_mnist_prepared.npz"

# podaci
TRAIN_CSV = DATA_RAW_DIR / "sign_mnist_train.csv"
TEST_CSV = DATA_RAW_DIR / "sign_mnist_test.csv"

IMG_SIZE = 28
NUM_CLASSES = 24
EXCLUDED_LABELS = {9, 25}  # J, Z

SEED = 42
VAL_SIZE = 0.15 # udeo trening skupa izdvojen za validaciju (stratifikovano)
BATCH_SIZE = 64

LETTERS = [chr(ord("A") + i) for i in range(26)]
LABEL_TO_LETTER = {i: LETTERS[i] for i in range(26) if i not in EXCLUDED_LABELS}
LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}


# Dataset klasa (identična onoj iz 02_priprema_podataka.ipynb)
class SignMnistDataset(Dataset):
    """Slike (N, 28, 28), vrednosti u [0, 1], i pripadajuće oznake."""

    def __init__(self, pixels, labels, transform=None):
        self.pixels = pixels.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.pixels[idx]).unsqueeze(0)  # (1, 28, 28)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_train_transform():
    """Ista augmentacija kao u 02_priprema_podataka.ipynb"""
    return transforms.RandomAffine(
        degrees=10,
        translate=(0.08, 0.08),
        scale=(0.9, 1.1),
    )

# Učitavanje pripremljenih podataka (sačuvanih iz 02_priprema_podataka.ipynb)
def load_prepared_arrays(npz_path=PROCESSED_NPZ):
    """Učitava numpy nizove sačuvane na kraju 02_priprema_podataka.ipynb.

    Vraća: train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels
    Pikseli su oblika (N, 28, 28), float32, normalizovani u [0, 1].
    Labele su celobrojne (0-25, bez 9 i 25).
    """
    if not Path(npz_path).exists():
        raise FileNotFoundError(
            f"Nije pronađen pripremljeni fajl: {npz_path}\n"
            f"Prvo pokreni 02_priprema_podataka.ipynb do kraja (uključujući "
            f"poslednju ćeliju koja poziva np.savez_compressed)."
        )

    data = np.load(npz_path)

    train_pixels, train_labels = data["train_pixels"], data["train_labels"]
    val_pixels, val_labels = data["val_pixels"], data["val_labels"]
    test_pixels, test_labels = data["test_pixels"], data["test_labels"]

    assert train_pixels.shape[1:] == (IMG_SIZE, IMG_SIZE), (
        f"Očekivan oblik (N,{IMG_SIZE},{IMG_SIZE}), dobijeno {train_pixels.shape}"
    )
    assert train_pixels.max() <= 1.0 + 1e-6, "Očekivani su normalizovani pikseli u [0,1]"

    return train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels


# pomoćna funkcija koja Kreira SignMnistDataset objekte (train/val/test) iz sačuvanih nizova
def build_datasets(npz_path=PROCESSED_NPZ, with_augmentation=True):
    train_pixels, train_labels, val_pixels, val_labels, test_pixels, test_labels = (
        load_prepared_arrays(npz_path)
    )

    train_transform = get_train_transform() if with_augmentation else None

    train_ds = SignMnistDataset(train_pixels, train_labels, transform=train_transform)
    val_ds = SignMnistDataset(val_pixels, val_labels)
    test_ds = SignMnistDataset(test_pixels, test_labels)

    return train_ds, val_ds, test_ds
