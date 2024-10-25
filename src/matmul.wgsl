@group(0)
@binding(0)
var<storage, read_write> state_buffer: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> weight_buffer: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> index_buffer: array<u32>;

fn matmul(column: u32, matrix_start: u32, input_vector_start: u32, output_vector_start: u32, output_size: u32) -> u32{
    var sum: u32 = 0;
    for (var i: u32 = 0; i < output_size; i++) {
        var matrix_index = i + column * n;
        sum += matrix_buffer[matrix_start + matrix_index] * input_vector_buffer[input_vector_start + i];
    }
    return sum;
}


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

   let matrix_start = index_buffer[0];
   let input_vector_start = index_buffer[1];
   let output_vector_start = index_buffer[2];
   let output_vector_end = index_buffer[3];

   var output_size : u32 = output_vector_end - output_vector_start + 1; // TODO check for off by one error.

    output_vector_buffer[output_vector_start + global_id.x] = matmul(global_id.x, matrix_start, input_vector_start, output_size);
}
