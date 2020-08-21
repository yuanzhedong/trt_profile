import numpy as np
from seg_dataset import SegmentationDataset

def test_segmentation_dataset_shape():
    size = 256
    num_block = 8
    dataset = SegmentationDataset(size = size)
    Xs, Ys = dataset.get_data(num_block)
    assert Xs.shape == (num_block, size, size, 3)
    assert Ys.shape == (num_block, size, size, 1)


def test_segmentation_dataset_value():
    size = 256
    dataset = SegmentationDataset(size = size)
    Xs, Ys = dataset.get_data(1)
    print(np.unique(Xs))
    assert len(np.unique(Xs)) == 2
    assert np.array_equal(Xs[0][:,:,0], Ys[0][:, :, 0])
