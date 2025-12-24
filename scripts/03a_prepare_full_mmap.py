"""
03a_prepare_full_mmap.py

OBJECTIVE:
Prepare the MS MARCO dataset (8.8 Million vectors) for indexing.
Instead of loading all data into RAM (which would crash the computer),
we merge all 45 .npy files into one giant binary file on the Hard Drive.
We then use "Memory Mapping" to let the OS handle loading only what is needed.

OUTPUT:
data/corpus_full_mmap.dat (~27 GB file)
"""

import numpy as np
import glob
import os

# --- Configuration ---
EMB_DIR = "data/embeddings"              # Where your .npy shards are
MMAP_FILENAME = "data/corpus_full_mmap.dat" # The output giant file
DIM = 768                                # MPNet dimension

def create_mmap():
    # 1. SCAN FILES
    # We need to know exactly how big the final dataset is before we start.
    files = sorted(glob.glob(os.path.join(EMB_DIR, "*.npy")))
    if not files:
        print("❌ No embedding files found!")
        return

    print("Scanning files to calculate total size...")
    total_count = 0
    
    # Loop through all files just to read their metadata (shape)
    for f in files:
        # mmap_mode='r' reads just the header info, not the data. Instant.
        shape = np.load(f, mmap_mode='r').shape
        total_count += shape[0]
    
    print(f"📉 Total Vectors Found: {total_count:,}")
    # Calculate expected size in GB: Vectors * Dims * 4 bytes (float32)
    print(f"💾 Expected File Size: {total_count * DIM * 4 / (1024**3):.2f} GB")

    # 2. ALLOCATE DISK SPACE
    print(f"Creating memory-mapped file at {MMAP_FILENAME}...")
    # np.memmap creates a placeholder file on disk of the exact size we need.
    # mode='w+' means "Create new or overwrite existing"
    fp = np.memmap(MMAP_FILENAME, dtype='float32', mode='w+', shape=(total_count, DIM))
    
    # 3. FILL WITH DATA (Streaming)
    # We load one small file at a time, copy it to the big file, then clear RAM.
    current_idx = 0
    for f in files:
        print(f"  Copying {os.path.basename(f)}...")
        
        # Load the actual data for this shard into RAM
        data = np.load(f) 
        count = data.shape[0]
        
        # Assign it to the specific slice of the big file on disk
        fp[current_idx : current_idx + count] = data
        
        # Move the pointer forward
        current_idx += count
        
        # CRITICAL: Delete the shard from RAM immediately to prevent memory leaks
        del data 
        
    # 4. FLUSH TO DISK
    # Ensure all operating system buffers are written to the physical drive
    fp.flush() 
    print("✅ Full corpus prepared on disk. You are ready to build indexes.")

if __name__ == "__main__":
    create_mmap()