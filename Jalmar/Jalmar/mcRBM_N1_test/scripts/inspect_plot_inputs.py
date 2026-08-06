import numpy as np
import h5py

vis = r"C:\Users\jalma\OneDrive - HAN\stage_donders\new_code_placeholder\plots\visualize_N1_V2_train\visData.npz"
flt = r"C:\Users\jalma\OneDrive - HAN\stage_donders\features\processing_test\raw_full\sleep_features_full_selection_raw_artifact_filtered.h5"

print('visData.npz:')
arr = np.load(vis, allow_pickle=True)
print('  keys:', list(arr.keys()))
print('  data shape:', arr['data'].shape, 'dtype:', arr['data'].dtype)
obs = arr['obsKeys']
print('  obsKeys type:', type(obs), 'shape:', obs.shape, 'dtype:', obs.dtype)
# Print a compact summary of obsKeys contents
flat = np.asarray(obs).reshape(-1)
print('  obsKeys first 10:', flat[:10])

print('\nArtifact-filtered HDF5:')
with h5py.File(flt, 'r') as f:
    subs = [k for k in f.keys() if k not in ('stats', 'intermediate')]
    print('  subjects:', len(subs))
    uniq = set()
    has5 = False
    total = 0
    for s in subs:
        labels = np.asarray(f[s]['scores'][:]).reshape(-1)
        valid = labels[~np.isnan(labels)]
        total += valid.shape[0]
        u = np.unique(valid)
        uniq.update(u.tolist())
        if 5 in u:
            has5 = True
    print('  total labels:', total)
    print('  unique labels:', sorted(int(x) for x in uniq))
    print('  has label 5:', has5)
