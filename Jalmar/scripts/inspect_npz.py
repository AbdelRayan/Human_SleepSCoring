import numpy as np
from pathlib import Path

p = Path(r"C:/Users/jalma/OneDrive - HAN/stage_donders/features/N1_selection_npz/sleep_features_N1_selection_test.npz")
if not p.exists():
    raise SystemExit(f"File not found: {p}")

data = np.load(p, allow_pickle=True)
print('files:', data.files)
for k in data.files:
    arr = data[k]
    print(f"\n=== {k} ===")
    print('shape:', arr.shape)
    print('dtype:', getattr(arr, 'dtype', type(arr)))
    if arr.size == 0:
        print('(empty)')
        continue
    # show small sample
    print('first rows:\n', arr[:5])
    if arr.ndim == 2:
        ncols = arr.shape[1]
        for i in range(ncols):
            col = arr[:, i]
            try:
                uniq = np.unique(col)
                print(f'col {i}: dtype={col.dtype}, unique_count={uniq.size}, sample={uniq[:10]}')
            except Exception as e:
                print('col', i, 'error', e)
    else:
        try:
            uniq = np.unique(arr)
            print('unique_count:', uniq.size, 'sample:', uniq[:10])
        except Exception as e:
            print('unique error', e)

print('\nInspection complete')
