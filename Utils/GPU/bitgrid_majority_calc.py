import cupy as cp
import cupyx

#################################
# To do:
# Convert bitgrid majority calculator to cuda code
#
#################################

class Bitgrid_Majority:
    
    def __init__(self, patch_class_array, num_classes):

        frames, rows, patch_h, cols, patch_w = patch_class_array.shape

        self.rows = rows
        self.cols = cols
        self.cells = rows * cols

        self.frames = frames
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.samples = frames * patch_h * patch_w

        self.num_classes = num_classes

        # ---- Persistent GPU buffers (allocated ONCE) ----

        # Count matrix: (cells, classes)


    def compute(self, patch_class_array):

        # ---- Move to GPU ----
        patch_gpu = cp.asarray(patch_class_array, dtype=cp.int32)

        # (frames, rows, patch_h, cols, patch_w)
        patch_gpu = patch_gpu.transpose(1, 3, 0, 2, 4)
        # (rows, cols, frames, patch_h, patch_w)

        rows, cols, frames, ph, pw = patch_gpu.shape

        # ---- Flatten into (cells, samples) ----
        flat = patch_gpu.reshape(
            rows * cols,
            frames * ph * pw
        )

        # ---- Allocate count matrix ----
        counts = cp.zeros((rows * cols, self.num_classes), dtype=cp.int32)

        # ---- Atomic scatter-add ----
        self.cell_indices = cp.repeat(
            cp.arange(rows * cols, dtype=cp.int32),
            flat.shape[1]
        )
        self.class_indices = flat.ravel()

        cupyx.scatter_add(
            counts,
            (self.cell_indices, self.class_indices),
            1
        )

        # ---- Majority vote ----
        majority = cp.argmax(counts, axis=1)

        return cp.asnumpy(majority.reshape(rows, cols))
