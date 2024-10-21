use std::ops::{Index, IndexMut, Range, Drop, Deref, DerefMut};
use std::clone::Clone;

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

/// the type for KV-cache et value-cache
pub struct Cache<'a>{
    gpu_context: &'a WgpuContext<'a>,
}

impl <'a> Cache<'a>{
    pub fn new(gpu_context: &'a WgpuContext, size: usize) -> Cache<'a>{
        // les vecteurs sont initializés à 0.0
        todo!()
    }
}

impl <'a> Index<Range<usize>> for Cache<'a> {
    type Output = DropGuard<'a>;

    fn index(&self, range: Range<usize>) -> &Self::Output {
        todo!()
    }
}

impl <'a> IndexMut<Range<usize>> for Cache<'a> {
//    type Output = DropGuard<'a>;

    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        todo!()
    }
}

pub struct DropGuard<'a>(Vector<'a>);

//The dropguard is there to make sure that the vectors taken from the kv and value caches are updated on the GPU side when being dropped
impl <'a>Drop for DropGuard<'a> {
    fn drop(&mut self) {
        todo!()
    }
}

impl <'a> Deref for DropGuard<'a> {
    type Target=Vector<'a>;

    // Required method
    fn deref(&self) -> &Self::Target{
        &self.0
    }
}

impl <'a> DerefMut for DropGuard<'a> {
    // Required method
    fn deref_mut(&mut self) -> &mut Vector<'a>{
        &mut self.0
    }
}

pub struct Vector<'a>{
    gpu_context: &'a WgpuContext<'a>,
    range_start: u32,
    range_end: u32,
    tensor_content: TensorContent,
}

impl <'a> Vector<'a>{

    pub fn new(gpu_context: &'a WgpuContext, size: usize) -> Vector<'a>{
        // les vecteurs sont initializés à 0.0
        todo!()
    }

    /// Get immutable access to a slice of the data in the tensor
    /// If the data is on the GPU, fetch it first
    pub fn data(&self)-> &[f32]{
        todo!()
    }

    /// Get immutable access to a slice of the data in the tensor
    /// If the data is on the GPU, fetch it first
    pub fn data_mut(&mut self)-> &mut [f32]{
        todo!()
    }
}

pub struct Matrix<'a>{
    gpu_context: &'a WgpuContext<'a>,
    range_start: u32,
    range_end: u32,
    tensor_content: TensorContent,
}

impl <'a> Matrix<'a>{

    pub fn new(gpu_context: &'a WgpuContext, range_start: u32, range_end: u32) -> Matrix<'a>{
        todo!()
    }

    /// Get immutable access to a slice of the data in the tensor
    /// Since the data is immutable, we get it directly from the original data located on the CPU side, we don't have to load it from the GPU
    pub fn data(&self) -> &[f32]{
        todo!()
    }
}

/// Weights are bags of matrices
pub struct Weights<'a>{
    gpu_context: &'a WgpuContext<'a>,
}

impl <'a> Weights<'a>{
    
    pub fn new(gpu_context: &'a WgpuContext<'a>, begining: u32, end:u32) -> Weights<'a>{
        todo!()
    }

    pub fn as_matrix(&self) -> &Matrix<'a>{
        todo!()
    }
}

/// Since Weight are immutable, we can do a shallow clone
impl <'a> Clone for Weights<'a>{
    fn clone(&self) -> Self{
      todo!()
    }
}

impl <'a> Index<Range<usize>> for Weights<'a> {
    type Output = Matrix<'a>;

    fn index(&self, range: Range<usize>) -> &Self::Output {
        todo!()
    }
}


pub fn matmul<'a>(output: &mut Vector<'a>, input: &Vector<'a>, matrix: &Matrix<'a>){
    assert!(std::ptr::eq(input.gpu_context, matrix.gpu_context), "Input vector and matrix must live in the same GPU context");
    todo!()
}

