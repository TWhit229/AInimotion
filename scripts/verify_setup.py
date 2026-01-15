"""Quick verification script for training setup."""
import torch
import multiprocessing

def main():
    # Enable cuDNN benchmark for max performance
    torch.backends.cudnn.benchmark = True

    from ainimotion.data.dataset import TripletDataset

    print("Testing triplet dataset loading...")
    dataset = TripletDataset("D:/Triplets", augment=False, crop_size=(256, 256))
    print(f"Dataset size: {len(dataset)} triplets")

    # Test loading first sample
    sample = dataset[0]
    print(f"Sample frames: frame1={sample['frame1'].shape}, frame2={sample['frame2'].shape}, frame3={sample['frame3'].shape}")

    # Test dataloader with high parallelism
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=24,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    print(f"DataLoader: {len(dataloader)} batches, batch_size=24, num_workers=12")

    # Load first batch to confirm GPU transfer works
    print("Loading first batch...")
    batch = next(iter(dataloader))
    print(f"First batch loaded: inputs={batch['inputs'].shape}")

    print("Moving to GPU...")
    inputs = batch["inputs"].cuda(non_blocking=True)
    print(f"GPU batch shape: {inputs.shape}")
    print("✓ All tests passed! Training ready to go on RTX 5070 Ti.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
