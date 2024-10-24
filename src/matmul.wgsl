@group(0)
@binding(0)
var<storage, read_write> input_vector_buffer: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> matrix_buffer: array<u32>;

@group(0)
@binding(2)
var<storage, read_write> output_vector_buffer: array<u32>;

fn matmul(column: u32) -> u32{
    var n: u32 = 4;// arrayLength(input_vector_buffer);

    var sum: u32 = 0;

    for (var i: u32 = 0; i < n; i++) {
        var matrix_index = i + column * n;
        sum += matrix_buffer[matrix_index] * input_vector_buffer[i];
    }
    return sum;
}


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    output_vector_buffer[global_id.x] = matmul(global_id.x);
}
