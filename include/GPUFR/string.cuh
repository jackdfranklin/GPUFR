#include <string>

namespace cu_type
{
    template<int size_val>
    class string
    {
        private:
        char data[size_val];

        public:
        __host__ __device__ string()
        {

        }

        __host__ string(const std::string &r_val)
        {
            int len = size_val < r_val.size()? size_val : r_val.size();
            for (int i=0; i<len; i++)
            {
                data[i] = r_val[i];
            }
            data[len] = '\0';
        }

        template<int r_size>
        __host__ string(const char (&r_val)[r_size])
        {
            int len = size_val < r_size? size_val : r_size;
            for (int i=0; i<len; i++)
            {
                data[i] = r_val[i];
            }
            data[len] = '\0';
        }

        __host__ __device__ inline char& operator[](int index)
        {
            return data[index];
        }

        __host__ __device__ inline char operator[](int index) const
        {
            return data[index];
        }

        __host__ __device__ inline bool operator==(string r_val)
        {
            for (int i=0; i<size_val; i++)
            {
                if (data[i] != r_val[i])
                    return false;
                if (data[i] == '\0')
                    return true;
            }

            return false;
        }

        __host__ __device__ inline int size()
        {
            return size_val;
        }

        __host__ __device__ inline string<size_val> substr(int start)
        {
            string<size_val> result;
            for (int i=0; i<size_val-start; i++)
            {
                result[i] = data[i+start];
            }

            return result;
        }

        __host__ __device__ inline const char* c_str() const
        {
            return data;
        }

        __host__ inline void operator=(const std::string &r_val)
        {
            int len = size_val < r_val.size()? size_val : r_val.size();
            for (int i=0; i<len; i++)
            {
            data[i] = r_val[i];
            }
            data[len] = '\0';
        }

        template<int r_size>
        __host__ inline void operator=(const char (&r_val)[r_size])
        {
            int len = size_val < r_size? size_val : r_size;
            for (int i=0; i<len; i++)
            {
                data[i] = r_val[i];
            }
            data[len] = '\0';
        }
    };

    template<int m, int n>
    __host__ __device__ inline bool operator==(const string<m>& l_val, const char (&r_val)[n])
    {
        int len = m < n? m : n;
        for (int i=0; i<len; i++)
        {
            if (l_val[i] != r_val[i])
                return false;
        }

        return true;
    }

    template<int n>
    __host__ __device__ u32 strtou(const string<n>& in)
    {
        u32 result = 0;
        for (int i=0; i<n; i++)
        {
            if (in[i] == '\0')
            {
                return result;
            }
            result *= 10;
            result += (in[i] - '0');
        }
        return result;
    } 
}