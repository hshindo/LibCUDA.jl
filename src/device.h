template<typename T, int N>
struct Array {
    T *data;
    const int dims[N];
    const int strides[N];
    const bool contigious;
public:
    __device__ int length() {
        int n = dims[0];
        for (int i = 1; i < N; i++) n *= dims[i];
        return n;
    }
    __device__ T &operator[](int idx) { return data[idx]; }
    __device__ T &operator()(int idx) {
        if (contigious) return data[idx];
        if (N == 1) return data[idx*strides[0]];

        int ndIdx[N];
        idx2ndIdx(ndIdx, idx);
        return (*this)(ndIdx);
    }
    __device__ T &operator()(int idx0, int idx1) {
        int idx = idx0*strides[0] + idx1*strides[1];
        return data[idx];
    }
    __device__ T &operator()(int idx0, int idx1, int idx2) {
        int idx = idx0*strides[0] + idx1*strides[1] + idx2*strides[2];
        return data[idx];
    }
    __device__ T &operator()(int idx0, int idx1, int idx2, int idx3) {
        int idx = idx0*strides[0] + idx1*strides[1] + idx2*strides[2] + idx3*strides[3];
        return data[idx];
    }
    __device__ T &operator()(const int ndIdx[N]) {
        int idx = 0;
        for (int i = 0; i < N; i++) {
            idx += ndIdx[i] * strides[i];
        }
        return data[idx];
    }
    __device__ void idx2ndIdx(int ndIdx[N], int idx) {
        ndIdx[0] = 1;
        for (int i = 1; i < N; i++) ndIdx[i] = ndIdx[i-1] * dims[i-1];

        int temp = idx;
        for (int i = N-1; i >= 0; i--) {
            int a = temp / ndIdx[i];
            temp -= a * ndIdx[i];
            ndIdx[i] = a;
        }
        return;
    }
};

template<int N>
struct Dims {
    const int data[N];
public:
    __device__ int operator[](int idx) { return data[idx]; }
};
