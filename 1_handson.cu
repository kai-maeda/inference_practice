#include <cuda_runtime.h>
#include <math.h>
//Write a Naive Matmul, Tiled matmul, profile both, compute arithmetic intensity, plot on a roofline

//(MxN)(NxK) = (MxK)
__global__ void matmul(float *A, float *B, float* C, int M, int N, int K) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float Pvalue = 0
        for(int n = 0; n < N; n++) {
            Pvalue += A[row * N + n] * B[n * K + col]
        }
        C[row * K + col] = Pvalue
    }

}
__global__ void tiled(float *A, float *B, float* C, int M, int N, int K, int TILE_WIDTH) {
    __shared__ float A_s[TILED_WIDTH][TILED_WIDTH];
    __shared__ float B_s[TILED_WIDTH][TILED_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = bx * TILE_WIDTH + tx;
    int col = by * TILE_WIDTH + ty;

    float Pvalue  = 0;

    for (int ph = 0; ph < ceil((float)N / TILE_WIDTH); ph++) {
        if (row < M && (ph * TILE_WIDTH + tx) < N) {
            A_s[ty][tx] = A[row*N  + (ph * TILE_WIDTH + tx)];
        } else {
            A_s[ty][tx] = 0.0;
        }

        if ((ph * TILE_WIDTH + ty) < N && col < K) {
            B_s[ty][tx] = B[(ph * TILE_WIDTH + ty) * K + col];
        } else{
            B_s[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            Pvalue += A_s[ty][i] * B_s[i][tx];
        }
        
        __syncthreads();
    }
    if(row < M && col < K) {
        C[row* K + col] =Pvalue;
    }
}


int main() {
    int M = 150; 
    int N = 300; 
    int K = 125;

    std::vector<float> h_C_gpu(M * K, 0.0f);

    float *d_A, *d_B, *d_C;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    
    dim3 gridDim(ceil((float)K / TILE_WIDTH), ceil((float)M / TILE_WIDTH));

    std::cout << "Launching Kernel with Grid (" << gridDim.x << ", " << gridDim.y << ") "
              << "and Block (" << blockDim.x << ", " << blockDim.y << ")\n";

    tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    return 0;
}