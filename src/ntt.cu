#include "GPUFR/ntt.cuh"

#define PRIME 105097513 // Must be less than (max unsigend) / 2

__device__ int bit_reverseal(int index, int log_2n, int required_threads)
{
    int i = index;

    int rev_index = 0;
    if (index < required_threads)
    {
        for (unsigned int j = 0; j < log_2n; j++) {
            rev_index <<= 1;       // Shift result left
            rev_index |= (i & 1);  // Set the lowest bit of i in result
            i >>= 1;            // Shift i right
        }
    }

    return rev_index;
}

__global__ void re_order(u32* array, int log_2n, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    u32 tmp;
    if (idx < required_threads)
    {
        int desired_index = bit_reverseal(idx, log_2n, required_threads);

        tmp = array[desired_index]; // Try to do more contiguously
    }
    __syncthreads();

    if (idx < required_threads)
    {
        array[idx] = tmp;
    }
}

__global__ void re_order_2(u32* array, u32* out, int log_2n, int required_threads)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < required_threads)
    {
        int desired_index = bit_reverseal(idx, log_2n, required_threads);

        u32 tmp = array[desired_index]; // Try to do more contiguously
        out[idx] = tmp;
    }
}

__global__ void NTT_level(u32* array, u32* out, u32 w, int level, int size, int required_threads, u32 prime)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    
    if(idx < required_threads)
    {
        int sub_step = 1<<(level-1);
        int read_loc = (idx%sub_step) + (1<<(level)) * (idx / sub_step); // i%l + 2^(l+1) * floor(i/l)
        int w_exp_1 = ((size*read_loc) / (1<<level)) % size; // floor(N*i / 2^(l))
        int w_exp_2 = ((size*(read_loc + sub_step)) / (1<<level)) % size; // floor(N*i / 2^(l))
        
        u32 in_1 = array[read_loc];
        u32 in_2 = array[read_loc+sub_step];

        printf("idx: %i read_loc: %i sub_step: %i exp1: %i exp2 %i \n", idx, read_loc, sub_step, w_exp_1, w_exp_2);

        u32 twiddle_1 = ff_pow(w, w_exp_1, prime); // Need to precompute
        u32 twiddle_2 = ff_pow(w, w_exp_2, prime); // Need to precompute

        u32 out_1 = ff_add( in_1, ff_multiply(in_2, twiddle_1, prime), prime);
        u32 out_2 = ff_add( in_1, ff_multiply(in_2, twiddle_2, prime), prime);

        out[read_loc] = out_1;
        out[read_loc+sub_step] = out_2;
    }
}

std::vector<u32> get_w(const std::string& filename, int index)
{
    std::ifstream file(filename); // Open the file
    std::vector<u32> numbers; // Vector to store the numbers
    std::string line;

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return numbers;
    }

    // Read a line from the file
    if (std::getline(file, line)) {
        std::stringstream ss(line); // Use stringstream to parse the line
        long long number;

        // Extract each number from the line
        while (ss >> number) {
            numbers.push_back(number);
        }
    }

    // Close the file
    file.close();
    
    return numbers;
}

void do_ntt(u32* &cu_array, u32* &cu_output, int arr_size, std::vector<u32> ws, u32 prime)
{
    int num_levels = log2(arr_size);
    printf("num_levels: %i \n", num_levels);

    u32 w = ws[num_levels];

    int required_threads = arr_size;
    int threadsPerBlock = required_threads>256? 256 : required_threads;
    int blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

    re_order_2<<<blocksPerGrid, threadsPerBlock>>>(cu_array, cu_output, num_levels, required_threads);
    std::swap(cu_array, cu_output);

    for (int level=1; level<= num_levels; level++)
    {
        required_threads = arr_size / 2;
        threadsPerBlock = required_threads>256? 256 : required_threads;
        blocksPerGrid = (required_threads + threadsPerBlock - 1) / threadsPerBlock;

        NTT_level<<<blocksPerGrid, threadsPerBlock>>>(cu_array, cu_output, w, level, arr_size, required_threads, prime);
        std::swap(cu_array, cu_output);
    }

    std::swap(cu_array, cu_output);
}