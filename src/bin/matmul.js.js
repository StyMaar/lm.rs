function matmul(matrix_buffer, input_vector_buffer, column){
    debugger;
    var n = input_vector_buffer.length;

    var sum = 0;
    var matrix_index;
    for (var i = 0; i < n; i++) {
        matrix_index = i + column * n;
        sum += matrix_buffer[matrix_index] * input_vector_buffer[i];
    }
    return sum;
}

function test(){
  
    let matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1];

//    Représente la matrice:
//    | 1 0 0|
//    | 0 1 0|
//    | 0 0 1|
//    | 0 0 1|


    let vector = [
        // [1, 0, 0, 0],
        [2, 2, 2, 2],
        [1, 2, 3, 4],
        [1, 4, 9, 16],
    ];

    for(input of vector){
        let out = [0, 0 ,0];

        for(let i = 0; i<3; i++){
            out[i] = matmul(matrix, input, i);
        }

        console.log(`Output of ${input} x ${matrix} = ${out}`);
    }
}