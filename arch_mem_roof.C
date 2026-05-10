
//1a.
__global__ void foo_kernel(float* A,float* B, float* C, unsigned int M, unsigned K, unsigned int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    // unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    //(MxK) dot (K by N) = (M by N)
    if (row < M) {
        for (i = 0; i < N; i++) {
            float Cvalue = 0
            for(j = 0; j < K; j++){
                //
                Cvalue += A[row*K+j]*B[j*N + i]
            }
            C[row*N + i] = Cvalue
        }
    }
}

dim3 threadsPerBlock(16,24);
dim3 numBlocks((N+16-1)/16, (M + 24-1) / 24);
foo_kerenl<<<numBlocks, threadsPerBlock>>>(A_d,B_d, C_d, M,K,N)
//1b.
__global__ void foo_kernel(float* A,float* B, float* C, unsigned int M, unsigned K, unsigned int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    //(MxK) dot (K by N) = (M by N)
    if (col < N) {
        for (i = 0; i < M; i++) {
            float Cvalue = 0
            for(j = 0; j < K; j++){
                //
                Cvalue += A[i*K+j]*B[j*N + col]
            }
            C[i*N + col] = Cvalue
        }
    }
}

dim3 threadsPerBlock(16,24);
dim3 numBlocks((N+16-1)/16, (M + 24-1) / 24);
foo_kernel<<<numBlocks, threadsPerBlock>>>(A_d,B_d, C_d, M,K,N)
//2.
__global__ void foo_kernel(float* A,float* B, float* C, unsigned int Width) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    // unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    //(MxM) dot (M by 1) = (M by 1)

    if (row < Width) {
        float Avalue = 0
        for (j = 0; j < Width; j++) {
            Avalue += B[row*Width + j] + C[j]
            
        }
        A[row*Width] = Avalue
    }
}
//3a. threads per block = 16 x 32 
//3b. 304 x 160 = 48640
//3c. (299)/16 + 1 x 149/32 + 1
//3d. 150 x 300

//4a. 400 * 20 + 10
//4b. 500 * 10 + 20

//5. 400 * 500 * 5  + 400 * 20 + 10

//1. threads per block = 128, total threads =1024, blocks = 8
// threads per warp = 32
// a. 128/32 = 4
//b. 1024/32
//c. 3/4 per block and 8 blocks means 24  warps, 16, 100, 8/32,  96-->127 (24/32)
//d. 100%, 100%, 50%
//e. 3, 2 
//2. 2048
//3. 1 warp
//4. 17.08%
//5. Not a good idea, warp size depends on architecture. Volta onward now feature independent thread scheduling

//6. 512 threads per block, dont have to use all blocks!
//7. a (50%),b (50%),c (50%), d (100%),e (100%)
//8. a. We can hit 2048 threads good! b. not good!, c. not good! too many registers
//9. N cubed operations, too many threads per block! also very slow!

//1. No, each thread reads from global memory once
//2. Memory read access reduced by tile width

//3. read before write error and write before read error
//4. all threads have access to it, so you dont have to grab from global memory per thread
//5. 32x reduction
//6. 1000 *512 times because local variables are thread level
//7. 1000 times because shared variables are block level
//8. N times, N/T
//9. 130 GFLOPs/S  memory bound, 321ish so compute bound
//10a. 1 to 400, only works for block size  = 1 because threads are dependent on threads written by other threads
//10b. insert syncthreads between 10 and 11 because it is read after write
//11. 1024, 1024, 8, 8, 10 4*129, 10 FLOPS / (4 + 1+ 1) * 4 bytes = 10/24 FP/B
//12a. 

//1.
#define TILE_WIDTH  = 32
__global__ void mult(float* A, float*B, float*C, int Width) {
    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH]
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH]

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty
    int col = bx * TILE_WIDTH + tx
    float Pvalue = 0
    for (int ph = 0; ph < Width/TILE_WIDTH; ph++){
        A_s[ty][tx] = A[Row*Width + (ph* TILE_WIDTH + tx)]
        int b_row = ph * TILE_WIDTH + tx;
        int b_col = bx * TILE_WIDTH + ty;
        B_s[tx][ty] = B[b_col * Width  + b_row];
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += A_s[ty][k]*B_s[k][tx];
        }
        __syncthreads();
    }
    C[Row*Width + Col] = Pvalue


}

//2. multiple of 32
//3. coalesced, NA, coalesced, uncoalsced, NA, NA, coalesc, NA, uncoalesc
//4. 2 OP / (2 elmement by 4 bytes) =.25, (1024 + 1024) * 4 bytes = 8k Bytes : 1024 threads * 32 steps * 2 OP = 65K OP, thread coarsening reduces shared reads and instruction overhead