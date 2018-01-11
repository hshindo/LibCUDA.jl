template<typename T, int N>
struct Array {
    T *data;
    const int dims[N];
public:
    __device__ int length() {
        int n = dims[0];
        for (int i = 1; i < N; i++) n *= dims[i];
        return n;
    }
    __device__ T &operator[](int idx) { return data[idx]; }
    __device__ T &operator()(int idx0, int idx1) {
        int idx = idx0 + idx1*dims[0];
        return data[idx];
    }
    __device__ T &operator()(int idx0, int idx1, int idx2) {
        int idx = idx0 + idx1*dims[0] + idx2*dims[0]*dims[1];
        return data[idx];
    }
    __device__ T &operator()(int idx0, int idx1, int idx2, int idx3) {
        int idx = idx0 + idx1*dims[0] + idx2*dims[0]*dims[1] + idx3*dims[0]*dims[1]*dims[2];
        return data[idx];
    }
    __device__ void idx2ndIdx(int ndIdx[N], int idx) {
        int cumdims[N];
        cumdims[0] = 1;
        for (int i = 1; i < N; i++) cumdims[i] = cumdims[i-1] * dims[i-1];
        int temp = idx;
        for (int i = N-1; i >= 1; i--) {
            int k = temp / cumdims[i];
            ndIdx[i] = k;
            temp -= k * cumdims[i];
        }
        ndIdx[0] = temp;
        return;
    }
    __device__ T& operator()(int ndIdx[N]) {
        int idx = 0;
        int stride = 1;
        for (int d = 0; d < N; d++) {
            if (dims[d] > 1) idx += ndIdx[d] * stride;
            stride *= dims[d];
        }
        return data[idx];
    }
};

template<int N>
struct Dims {
    const int data[N];
public:
    __device__ int operator[](int idx) { return data[idx]; }
};
