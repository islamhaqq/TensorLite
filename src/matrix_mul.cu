#define TILE_WIDTH 16 // Define tile width for shared memory; used to break down matrices for efficient processing in blocks

__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int Width) {
    // Define the kernel for matrix multiplication; it's a __global__ function since it's called from host and runs on device

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // Allocate shared memory for storing sub-matrix of M, improves memory bandwidth.
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; // Allocate shared memory for storing sub-matrix of N, for the same reason.

    int bx = blockIdx.x; // Get the x-coordinate in the block grid, identifies which block in grid.
    int by = blockIdx.y; // Get the y-coordinate in the block grid, identifies which block in grid.
    int tx = threadIdx.x; // Get hte x-coordinate in the block, identifies which thread in block.
    int ty = threadIdx.y; // Get the y-coordinate in the block, identifies which thread in the block.

    // Identify the row and column of the P matrix element to work on.
    int Row = by * TILE_WIDTH + ty; // Calculate global row index for the elemnt of P being processed.
    int Col = bx * TILE_WIDTH + tx; // Calculate global column index for the element of P being processed.

    float Pvalue = 0; // Initialize the value for hte lement of the result matrix P to zero.

    // Loop over the M and N sub-matrices required to compute the P element.
    for (int m = 0; m < Width / TILE_WIDTH; ++m) {
        // Load M and N sub-matrices from device memory to shared memory; each thread loads one elemnt of each sub-matrix.
        Mds[ty][tx] = d_M[Row * Width + (m * TILE_WIDTH + tx)];
        Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
        __syncthreads(); // Ensure all threads in the block have finished loading their elements before computation begins.

        // Multiple the two matrices together; each thread computes one element of the block sub-matrix;
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads(); // Ensure all threads have finished computing their elements before loading new sub-matrices in the next iteration.
    }

    d_P[Row * Width + Col] = Pvalue; // Write the computed element to the device memory; each thread writes on element.
}
