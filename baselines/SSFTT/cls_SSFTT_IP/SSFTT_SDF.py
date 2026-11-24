"""
SSFTT_SDF.py
Adapts SSFTT to use the SuperSDF spatial_split_data2 evaluation protocol.
- Same train/val/test split as SuperSDF
- Same 10 MC runs with seeds mc*7+13
- Same metrics: OA, AA, kappa, per-class accuracy
- CPU-compatible for Windows Spyder

Place this file in the same folder as:
    SSFTT.py  (original SSFTT implementation)
    spatial_split_data2.py
    your .mat dataset files

Usage: runfile('SSFTT_SDF.py')
"""

import os
import time
import warnings
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# Windows Spyder fix — must be before any SSFTT import
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from SSFTTnet import SSFTTnet          # adjust class name if different in your file
from spatial_split_data2 import spatial_split_data2
mat = sio.loadmat('Indian_pines.mat')
print([k for k in mat.keys() if not k.startswith('_')])
# =========================================================================
# 0. CONFIGURATION — match SuperSDF settings exactly
# =========================================================================
HSI_FILE   = 'PaviaUF.mat'
DATASET_ID = 'Pavia'
N_MC       = 1
N_EPOCHS   = 10

DS = {
    'Indian':  dict(train_ratio=0.05, n_pca=30, patch=13,
                    img_key='indian_pines_corrected', gt_key='indian_pines_gt'),
    'Pavia':   dict(train_ratio=0.05, n_pca=30, patch=13,
                    img_key='paviaU', gt_key='paviaU_gt'),
    'Salinas': dict(train_ratio=0.01, n_pca=30, patch=13,
                    img_key='salinas_corrected', gt_key='salinas_gt'),
    'HongHu':  dict(train_ratio=0.02, n_pca=30, patch=13,
                    img_key='img', gt_key='gt'),
}
cfg = DS[DATASET_ID]

TRAIN_RATIO  = cfg['train_ratio']
N_PCA        = cfg['n_pca']
PATCH_SIZE   = cfg['patch']

# Training settings from SSFTT paper
N_EPOCHS     = 100
BATCH_SIZE   = 64
LR           = 5e-4
WEIGHT_DECAY = 5e-3
PATIENCE     = 20       # early stopping patience (epochs)

MIN_DIST     = 5        # matches SuperSDF
N_MC         = 10       # matches SuperSDF
RNG_SEED     = 50       # matches SuperSDF

# Windows CPU/GPU detection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cpu':
    print('Running on CPU — expect longer training times.')
    print('Tip: reduce N_EPOCHS to 50 for a quick test run.')


# =========================================================================
# 1. DATA LOADING AND PCA
# =========================================================================
def load_dataset(hsi_file, cfg):
    """Load .mat file and apply PCA."""
    mat = sio.loadmat(hsi_file)

    # Try configured keys first, then common alternatives
    img_keys = [cfg['img_key'], 'img', 'data', 'image', 'HSI']
    gt_keys  = [cfg['gt_key'],  'gt',  'label', 'labels', 'groundtruth']

    img = None
    for k in img_keys:
        if k in mat:
            img = mat[k].astype(np.float64)
            print(f'Loaded image from key: {k}')
            break
    if img is None:
        # Print available keys to help debug
        print(f'Available keys: {[k for k in mat.keys() if not k.startswith("_")]}')
        raise KeyError('Image key not found. Update img_key in DS config.')

    gt = None
    for k in gt_keys:
        if k in mat:
            gt = mat[k].astype(int)
            print(f'Loaded GT from key: {k}')
            break
    if gt is None:
        raise KeyError('GT key not found. Update gt_key in DS config.')

    rows, cols, bands = img.shape
    print(f'Image: {rows}×{cols}×{bands}  Classes: {gt.max()}')

    # Normalize each band to [0, 1]
    for b in range(bands):
        lo, hi = img[:,:,b].min(), img[:,:,b].max()
        if hi > lo:
            img[:,:,b] = (img[:,:,b] - lo) / (hi - lo)

    # PCA reduction
    n_pca = min(cfg['n_pca'], bands)
    X     = img.reshape(rows * cols, bands)
    pca   = PCA(n_components=n_pca, whiten=True, random_state=RNG_SEED)
    Xpca  = pca.fit_transform(X).reshape(rows, cols, n_pca).astype(np.float32)

    # Normalize PCA output to [0, 1]
    for c in range(n_pca):
        lo, hi = Xpca[:,:,c].min(), Xpca[:,:,c].max()
        if hi > lo:
            Xpca[:,:,c] = (Xpca[:,:,c] - lo) / (hi - lo)

    return Xpca, gt, rows, cols, n_pca


print(f'\n=== SSFTT — SuperSDF Protocol ===')
print(f'Dataset: {DATASET_ID}   File: {HSI_FILE}')
img_pca, gt, rows, cols, n_pca = load_dataset(HSI_FILE, cfg)
n_classes = int(gt.max())
print(f'PCA bands: {n_pca}  Classes: {n_classes}\n')


# =========================================================================
# 2. PATCH DATASET
# =========================================================================
class HSIPatchDataset(Dataset):
    """
    Returns (patch_tensor, label) where:
    patch_tensor : (n_pca, patch_size, patch_size) float32
    label        : int, 0-indexed
    pixel_indices: indices into the valid_indices array
    """
    def __init__(self, img_pca, gt, pixel_indices, valid_indices,
                 patch_size, rows, cols):
        self.patch_size = patch_size
        half = patch_size // 2

        # Reflect-pad image to handle border pixels
        self.img_pad = np.pad(
            img_pca,
            ((half, half), (half, half), (0, 0)),
            mode='reflect'
        )

        gt_flat = gt.flatten()
        self.patches = []
        self.labels  = []

        for idx in pixel_indices:
            flat_idx = valid_indices[idx]
            r = flat_idx // cols
            c = flat_idx %  cols
            patch = self.img_pad[r:r+patch_size, c:c+patch_size, :]
            self.patches.append(patch.copy())
            self.labels.append(int(gt_flat[flat_idx]) - 1)  # 0-indexed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.tensor(
            self.patches[i].transpose(2, 0, 1),   # (30, 13, 13)
            dtype=torch.float32
        ).unsqueeze(0)                             # (1, 30, 13, 13) ← per SAMPLE
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y


# =========================================================================
# 3. TRAIN / EVALUATE
# =========================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for x, y in loader:
        x    = x.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        all_pred.extend(pred.tolist())
        all_true.extend(y.numpy().tolist())
    return np.array(all_pred), np.array(all_true)


def compute_metrics(y_true, y_pred, n_cls):
    """OA, AA, kappa (%) and per-class accuracy (%)."""
    labels = np.arange(n_cls)
    cm     = confusion_matrix(y_true, y_pred, labels=labels)
    OA     = cm.diagonal().sum() / max(cm.sum(), 1) * 100
    ca     = np.array([
        cm[c, c] / max(cm[c].sum(), 1) * 100
        for c in range(n_cls)
    ])
    AA    = ca.mean()
    total = cm.sum()
    rs, cs = cm.sum(1), cm.sum(0)
    pe     = (rs * cs).sum() / max(total**2, 1)
    po     = OA / 100
    kappa  = (po - pe) / max(1 - pe, 1e-12) * 100
    return OA, AA, kappa, ca


# =========================================================================
# 4. MONTE-CARLO EVALUATION (10 runs — seeds match SuperSDF)
# =========================================================================
print(f'[Training] {N_MC} MC runs  patch={PATCH_SIZE}×{PATCH_SIZE}  '
      f'PCA={n_pca}  epochs={N_EPOCHS}\n')

valid_mask    = gt.flatten() > 0
valid_indices = np.where(valid_mask)[0]

OA_mc  = np.zeros(N_MC)
AA_mc  = np.zeros(N_MC)
kap_mc = np.zeros(N_MC)
CA_mc  = np.zeros((N_MC, n_classes))
time_mc = np.zeros(N_MC)

for mc in range(1, N_MC + 1):
    t0 = time.time()
    print(f'  MC {mc:2d}/{N_MC} ', end='', flush=True)

    # --- same seed formula as SuperSDF ---
    np.random.seed(mc * 7 + 13)
    torch.manual_seed(mc * 7 + 13)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed(mc * 7 + 13)

    # --- spatial split (placeholder 1-band, only indices matter) ---
    feat_placeholder = gt.reshape(rows, cols, 1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        (_, _, _, _, _, _, _,
         tr_idx, vl_idx, te_idx, _) = spatial_split_data2(
             feat_placeholder, gt, TRAIN_RATIO, MIN_DIST,
             visualize=False)

    # --- patch datasets ---
    ds_tr = HSIPatchDataset(img_pca, gt, tr_idx, valid_indices,
                            PATCH_SIZE, rows, cols)
    ds_vl = HSIPatchDataset(img_pca, gt, vl_idx, valid_indices,
                            PATCH_SIZE, rows, cols)
    ds_te = HSIPatchDataset(img_pca, gt, te_idx, valid_indices,
                            PATCH_SIZE, rows, cols)

    # num_workers=0 is mandatory on Windows Spyder
    ld_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=False)
    ld_vl = DataLoader(ds_vl, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=False)
    ld_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=0, pin_memory=False)

    # --- build SSFTT model ---
    # Adjust constructor arguments to match your SSFTT.py
    try:
        model = SSFTTnet(
            in_channels=n_pca,
            num_classes=n_classes,
            patch_size=PATCH_SIZE
        ).to(DEVICE)
    except TypeError:
        # Some SSFTT versions use different argument names
        try:
            model = SSFTTnet(
                num_classes=n_classes
            ).to(DEVICE)
        except Exception as e:
            print(f'\nSSFTTnet constructor failed: {e}')
            print('Check argument names in your SSFTT.py and update above.')
            raise

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    # --- training with early stopping on validation OA ---
    best_val_oa    = 0.0
    best_state     = None
    no_improve     = 0

    for epoch in range(1, N_EPOCHS + 1):
        train_epoch(model, ld_tr, optimizer, criterion, DEVICE)
        scheduler.step()

        # Check val every 5 epochs
        if epoch % 5 == 0 or epoch == N_EPOCHS:
            vl_pred, vl_true = evaluate(model, ld_vl, DEVICE)
            vl_oa = float((vl_pred == vl_true).mean() * 100)
            if vl_oa > best_val_oa:
                best_val_oa = vl_oa
                best_state  = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
                no_improve  = 0
            else:
                no_improve += 5
            if no_improve >= PATIENCE:
                break

    # --- load best model and test ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    te_pred, te_true = evaluate(model, ld_te, DEVICE)
    OA, AA, kappa, ca = compute_metrics(te_true, te_pred, n_classes)

    OA_mc[mc-1]  = OA
    AA_mc[mc-1]  = AA
    kap_mc[mc-1] = kappa
    CA_mc[mc-1]  = ca
    time_mc[mc-1] = time.time() - t0

    print(f'OA={OA:.2f}%  AA={AA:.2f}%  κ={kappa:.2f}%  '
          f'({time_mc[mc-1]:.1f}s)')

    # Free GPU memory between runs
    del model, best_state
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


# =========================================================================
# 5. RESULTS TABLE (same format as SuperSDF)
# =========================================================================
print(f'\n{"="*65}')
print(f'  SSFTT Results — {DATASET_ID}  ({N_MC} MC runs)')
print(f'  [SuperSDF spatial-split protocol, train={TRAIN_RATIO*100:.0f}%]')
print(f'{"="*65}')
print(f'{"Class":<6}  {"CA% mean ± std":<25}')
print('-' * 40)
for c in range(n_classes):
    print(f'  C{c+1:<4}  {CA_mc[:,c].mean():.2f} ± {CA_mc[:,c].std():.2f}')
print('-' * 40)
print(f'{"OA(%)":<6}  {OA_mc.mean():.2f} ± {OA_mc.std():.2f}')
print(f'{"AA(%)":<6}  {AA_mc.mean():.2f} ± {AA_mc.std():.2f}')
print(f'{"κ(%)":<6}  {kap_mc.mean():.2f} ± {kap_mc.std():.2f}')
print(f'{"Time":<6}  {time_mc.mean():.1f} ± {time_mc.std():.1f} s/run  '
      f'(total {time_mc.sum():.1f}s)')
print(f'{"Device":<6}  {DEVICE}')
print('='*65)