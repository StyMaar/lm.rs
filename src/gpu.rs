

pub struct WgpuContext<'a>{
    data: &'a [u8],
}

impl <'a> WgpuContext<'a> {
    pub fn new(data: &'a [u8]) -> WgpuContext<'a>{
        WgpuContext{
            data
        }
    }
}

enum TensorContent{
    InGPUMemory,
    InRam(Vec<u32>),
}

pub struct Tensor<'a>{
    gpu_context: &'a WgpuContext<'a>,
    range_start: u32,
    range_end: u32,
    tensor_content: TensorContent,
    // parent_tensor_lifetime: PhantomData<&'b Tensor<'a, 'b>>,
}

impl <'a> Tensor<'a>{

    pub fn new(gpu_context: &'a WgpuContext, range_start: u32, range_end: u32) -> Tensor<'a>{
        todo!()
    }

    /// Get immutable access to a slice of the data in the tensor
    /// If the data is on the GPU, fetch it first
    /// Begin and end represents the range of data inside this particular tensor
    pub fn get(&self, begin: usize, end: usize)-> &[u32]{
        todo!()
    }

    pub fn get_mut(&mut self, begin: usize, end: usize)-> &mut [u32]{
        todo!()
    }

    // fn child_tensor<'parent> (&'parent self, being: usize, end: usize)-> Tensor<'a, 'parent>{

    //     let tensor_content = match self.tensor_content{
    //         TensorContent::InGPUMemory => TensorContent::InGPUMemory,
    //         TensorContent::InRam(vec) => 
    //     };


    //     Tensor{
    //         gpu_context: &self.gpu_context,
    //         range_start: self.range_start,
    //         range_end: self.range_end,
    //         tensor_content: TensorContent::InGPUMemory, // todo!
    //         parent_tensor_lifetime: PhantomData,
    //     }
    // }
}

pub struct Vector<'a>(Tensor<'a>);

pub struct Matrix<'a>(Tensor<'a>);

pub fn matmul<'a>(input: &Vector<'a>, matrix: &Matrix<'a>) -> Vector<'a>{
    assert!(std::ptr::eq(input.0.gpu_context, matrix.0.gpu_context), "Input vector and matrix must live in the same GPU context");
    todo!()
}