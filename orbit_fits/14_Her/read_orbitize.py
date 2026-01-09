import h5py
import numpy as np

# Your HDF5 filename
hdf5_filename="14_Her_rv_hgca_posterior.h5"

print("Complete HDF5 file inspection:")
print("="*60)

with h5py.File(hdf5_filename,"r") as hf:
    print("\nAll datasets:")
    for key in hf.keys():
        dataset=hf[key]
        print(f"  {key}:")
        print(f"    - shape: {dataset.shape}")
        print(f"    - dtype: {dataset.dtype}")
        print(f"    - size: {dataset.size}")

    print("\nAll attributes:")
    for attr in hf.attrs.keys():
        print(f"  {attr}: {hf.attrs[attr]}")

    # Get actual array sizes
    post_shape=hf['post'].shape
    lnlike_shape=hf['lnlike'].shape
    theta_shape=hf['theta'].shape

    print("\n"+"="*60)
    print("Size comparison:")
    print(f"  post:   {post_shape[0]} samples × {post_shape[1]} parameters")
    print(f"  lnlike: {lnlike_shape[0]} values")
    print(f"  theta:  {theta_shape[0]} values")

    print("\n"+"="*60)
    print("ISSUE IDENTIFIED:")
    print(f"  'post' has {post_shape[0]} rows")
    print(f"  'lnlike' has {lnlike_shape[0]} rows")
    print(f"  These don't match! Difference: {lnlike_shape[0]-post_shape[0]}")