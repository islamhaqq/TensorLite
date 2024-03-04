__global__ void vector_add_kernel(float *out, const float *a, const float *b, int n) { // Kernel function
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Calculate the index of the current thread
    if (index < n) { // Ensure the current thread is within the array bounds
        out[index] = a[index] + b[index]; // Perform the vector addition
    }
}

extern "C" void vector_add(float *out, const float *a, const float *b, int n) {
    float *d_a, *d_b,  *d_out;
    size_t size = n * sizeof(float);

    // Allocate memory on the device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_out, size);

    // Copy inputs to the device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    vector_add_kernel<<<numBlocks, blockSize>>>(d_out, d_a, d_b, n);

    // Copy result back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}
