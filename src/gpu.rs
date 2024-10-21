use std::ops::{Index, IndexMut, Range, Drop};

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

pub struct Vector<'a>{
    gpu_context: &'a WgpuContext<'a>,
    range_start: u32,
    range_end: u32,
    tensor_content: TensorContent,
}

impl <'a> Vector<'a>{

    pub fn new(gpu_context: &'a WgpuContext, size: usize) -> Tensor<'a>{
        // les vecteurs sont initializés à 0.0
        todo!()
    }

    /// Get immutable access to a slice of the data in the tensor
    /// If the data is on the GPU, fetch it first
    /// Begin and end represents the range of data inside this particular tensor
    pub fn get(&self, begin: usize, end: usize)-> &[u32]{
        todo!()
        // Est-ce que ça sert à quelque chose ?
    }

    pub fn data(&self)-> &[u32]{
        todo!()
    }

    
    pub fn data_mut(&mut self)-> &mut [u32]{
        todo!()
    }

}

impl <'a> IndexMut<Range<u32>> for Vector<'a> {
    type Output = SubVector<'a>;

    fn index_mut(&mut self, range: Range<u32>) -> &Self::Output {
        todo!()
    }
}

pub struct SubVector<'a>{
    gpu_context: &'a WgpuContext<'a>,
    range_start: u32,
    range_end: u32,
    parent_vector: &'a Vector<'a>,
}

impl <'a> SubVector<'a>{
    pub fn data_mut(&mut self)-> &mut [u32]{
        todo!()
    }
}

impl Drop for SubVector {
    fn drop(&mut self) {
        todo!()
    }
}


pub struct Matrix<'a>{
    gpu_context: &'a WgpuContext<'a>,
    range_start: u32,
    range_end: u32,
    tensor_content: TensorContent,
    // parent_tensor_lifetime: PhantomData<&'b ImmutableTensor<'a, 'b>>,
}

// le type ImmutbleTensor ne sert à rien, ça devrait directement être implémenté pour le type `Matrix`
impl <'a> Matrix<'a>{

    pub fn new(gpu_context: &'a WgpuContext, range_start: u32, range_end: u32) -> ImmutableTensor<'a>{
        todo!()
    }

    /// Get immutable access to a slice of the data in the tensor
    /// Since the data is immutable, we get it directly from the original data located on the CPU side, we don't have to load it from the GPU
    pub fn data(&self) -> &[u32]{
        todo!()
    }

}

/// weights are bags of matrices
pub struct Weights<'a>{
    gpu_context: &'a WgpuContext<'a>,
};

impl <'a> Weights<'a>{
    
    fn new(gpu_context: &'a WgpuContext<'a>, begining: u32, end:u32) -> Weights<'a>{
        todo!()
    }

}

impl <'a> Index<Range<u32>> for Weights<'a> {
    type Output = Matrix<'a>;

    fn index(&self, range: Range<u32>) -> &Self::Output {
        todo!()
    }
}


pub fn matmul<'a, V: 'a + >(output: &mut Vector<'a>, input: &Vector<'a>, matrix: &Matrix<'a>){
    assert!(std::ptr::eq(input.0.gpu_context, matrix.0.gpu_context), "Input vector and matrix must live in the same GPU context");
    todo!()
}


pub fn matmul_s<'a, V: 'a + >(output: &mut SubVector<'a>, input: &Vector<'a>, matrix: &Matrix<'a>){
    assert!(std::ptr::eq(input.0.gpu_context, matrix.0.gpu_context), "Input vector and matrix must live in the same GPU context");
    todo!()
}
