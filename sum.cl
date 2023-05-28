// sum all values in input.
__kernel void sum(__global int *input, __global int *output, __local int *tmp, int inputSize) {

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    tmp[local_id] = global_id < inputSize ? input[global_id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // parallel reduction
    for (unsigned int stride = group_size >> 1; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            tmp[local_id] += tmp[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write to global mem
    if (local_id == 0) {
        output[group_id] = tmp[0];
    }
}