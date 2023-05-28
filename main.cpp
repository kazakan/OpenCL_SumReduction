#include "main.h"

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)

/*
 * Example of sum 1 to 1000
 * Expected result is "Result = 500500"
 */
int main() {
    const size_t N = 1000;

    const size_t group_size = 64;
    const int n_groups = (N + group_size - 1) / group_size; // ceil(N/group_size)
    const size_t global_size = n_groups * group_size;       // global size must be divisible by group_size (local)

    // init values
    cl_int *input = new cl_int[N];
    for (int i = 0; i < N; ++i) {
        input[i] = i + 1;
    }
    cl_int *output = new cl_int[n_groups];

    // Read .cl file
    ifstream ifs("../sum.cl");
    string sourcestr((istreambuf_iterator<char>(ifs)), (istreambuf_iterator<char>()));

    try {
        // init opencl
        vector<cl::Platform> platformList;
        cl::Platform::get(&platformList);

        cl_context_properties cprops[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platformList[0])(),
            0};

        cl::Context ctx(CL_DEVICE_TYPE_GPU, cprops);

        vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
        cl::CommandQueue queue(ctx, devices[0], 0);

        // create input & output buffers
        cl::Buffer input_buf = cl::Buffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_int), input);
        cl::Buffer output_buf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, n_groups * sizeof(cl_int), output);

        // create kernel from source code
        cl::Program::Sources sources;
        sources.push_back(sourcestr);
        cl::Program program(ctx, sources);
        program.build(devices);

        cl::Kernel kernel(program, "sum");

        // set kernel arg
        kernel.setArg(0, input_buf);
        kernel.setArg(1, output_buf);
        kernel.setArg(2, sizeof(cl_int) * group_size, NULL); // local
        kernel.setArg(3, N);

        // launch kernel
        // params =  kernel, offset, global work size, local work size
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(group_size)); 

        // get output
        cl_int *out = (cl_int *)queue.enqueueMapBuffer(output_buf, CL_TRUE, CL_MAP_READ, 0, n_groups * sizeof(cl_int));

        // Compute sum of each local work results
        int sum = 0;
        for (int i = 0; i < n_groups; ++i) {
            sum += out[i];
        }

        cout << "Result = " << sum << endl;

        // cleanup
        delete[] input;
        delete[] output;

        queue.enqueueUnmapMemObject(
            output_buf,
            (void *)out);

    } catch (exception err) {
        std::cerr
            << "ERROR: "
            << err.what()
            << std::endl;
    }

    return 0;
}