"""Our own GPU kernels (Triton). Each one computes the same function as the kernel it replaces and is
verified against it and against the float64 reference before use."""
