namespace cu_type
{
    template<typename T>
    class stack
    {
        private:
        T* ptr_top;
        T* ptr_bottom;

        public:

        __device__ stack(u32* stack_allocation, int thread_id, int max_stack)
        {
            // ptr_bottom = &data;
            ptr_bottom = stack_allocation + thread_id*max_stack;
            ptr_top = ptr_bottom-1;
        }

        __device__ void push(T val)
        {
            ptr_top += 1;
            *ptr_top = val;
        }

        __device__ void  pop()
        {
            ptr_top -= 1;
        }

        __device__ T top()
        {
            return *ptr_top;
        }
    };
}