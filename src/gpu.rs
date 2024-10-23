use std::clone::Clone;
use std::ops::{Deref, DerefMut, Drop, Index, IndexMut, Range};
use std::cmp;

pub struct WgpuContextBuilder{
//   data: &'a [u8],
   state_buffer_size: usize,
   temporary_buffer_size:usize,
}

impl WgpuContextBuilder{

   pub fn new() -> WgpuContextBuilder{
      WgpuContextBuilder{
//         data,
         state_buffer_size: 0,
         temporary_buffer_size: 0,
      }
   }

   pub fn make_state(&mut self, size: usize)-> GpuIndex{
      let index_begin = self.state_buffer_size;
      let index_end = self.state_buffer_size + size;

      self.state_buffer_size += size;
      self.temporary_buffer_size = cmp::max(self.temporary_buffer_size, size);
      
      GpuIndex{
         begin: index_begin,
         end: index_end,
      }
   }

   // create the GPU context
   pub fn finalize<'a>(self, data: &'a [u8])-> WgpuContext<'a>{
      todo!();
   }
}

pub struct GpuIndex{
   begin:usize,
   end: usize,
}

pub struct WgpuContext<'a> {
    pub data: &'a [u8],
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    state_buffer: wgpu::Buffer, // the buffer where the transformer state is stored
    weights_buffer: wgpu::Buffer, // the buffer for the transformer weights
    temporary_buffer: wgpu::Buffer, // buffer where the output of the multiplication is stored temporarily
    output_staging_buffer: wgpu::Buffer, // the buffer to send value back to the CPU
}

//impl WgpuContext {
//    async fn new(data: &'a [u8]) -> WgpuContext {

//        let instance = wgpu::Instance::default();
//        let adapter = instance
//            .request_adapter(&wgpu::RequestAdapterOptions::default())
//            .await
//            .unwrap();
//        let (device, queue) = adapter
//            .request_device(
//                &wgpu::DeviceDescriptor {
//                    label: None,
//                    required_features: wgpu::Features::empty(),
//                    required_limits: wgpu::Limits::downlevel_defaults(),
//                    memory_hints: wgpu::MemoryHints::Performance,
//                },
//                None,
//            )
//            .await
//            .unwrap();
//
//        // Our shader, kindly compiled with Naga.
//        let shader = device.create_shader_module(wgpu::include_wgsl!("gpu/shader.wgsl"));
//
//        let input_vector_buffer = device.create_buffer(&wgpu::BufferDescriptor {
//            label: Some("Vector Buffer"),
//            mapped_at_creation: false,
//            size: (input_size * 4usize) as wgpu::BufferAddress,
//            usage: wgpu::BufferUsages::STORAGE
//                | wgpu::BufferUsages::COPY_DST
//                | wgpu::BufferUsages::COPY_SRC,
//        });
//
//        let matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//            label: Some("Matrix Buffer"),
//            contents: bytemuck::cast_slice(&matrix),
//            usage: wgpu::BufferUsages::STORAGE
//                | wgpu::BufferUsages::COPY_DST
//                | wgpu::BufferUsages::COPY_SRC,
//        });
//
//        let output_vector_buffer = device.create_buffer(&wgpu::BufferDescriptor {
//            label: Some("Vector Buffer"),
//            size: (output_size * 4usize) as wgpu::BufferAddress,
//            mapped_at_creation: false,
//            usage: wgpu::BufferUsages::STORAGE
//                | wgpu::BufferUsages::COPY_DST
//                | wgpu::BufferUsages::COPY_SRC,
//        });
//
//        let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
//            label: None,
//            size: (output_size * 4usize) as wgpu::BufferAddress,
//            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
//            mapped_at_creation: false,
//        });
//
//        // This can be though of as the function signature for our CPU-GPU function.
//        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
//            label: None,
//            entries: &[
//                wgpu::BindGroupLayoutEntry {
//                    binding: 0,
//                    visibility: wgpu::ShaderStages::COMPUTE,
//                    ty: wgpu::BindingType::Buffer {
//                        ty: wgpu::BufferBindingType::Storage { read_only: false },
//                        has_dynamic_offset: false,
//                        // Going to have this be None just to be safe.
//                        min_binding_size: None,
//                    },
//                    count: None,
//                },
//                wgpu::BindGroupLayoutEntry {
//                    binding: 1,
//                    visibility: wgpu::ShaderStages::COMPUTE,
//                    ty: wgpu::BindingType::Buffer {
//                        ty: wgpu::BufferBindingType::Storage { read_only: false },
//                        has_dynamic_offset: false,
//                        // Going to have this be None just to be safe.
//                        min_binding_size: None,
//                    },
//                    count: None,
//                },
//                wgpu::BindGroupLayoutEntry {
//                    binding: 2,
//                    visibility: wgpu::ShaderStages::COMPUTE,
//                    ty: wgpu::BindingType::Buffer {
//                        ty: wgpu::BufferBindingType::Storage { read_only: false },
//                        has_dynamic_offset: false,
//                        // Going to have this be None just to be safe.
//                        min_binding_size: None,
//                    },
//                    count: None,
//                },
//            ],
//        });
//        // This ties actual resources stored in the GPU to our metaphorical function
//        // through the binding slots we defined above.
//        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
//            label: None,
//            layout: &bind_group_layout,
//            entries: &[
//                wgpu::BindGroupEntry {
//                    binding: 0,
//                    resource: input_vector_buffer.as_entire_binding(),
//                },
//                wgpu::BindGroupEntry {
//                    binding: 1,
//                    resource: matrix_buffer.as_entire_binding(),
//                },
//                wgpu::BindGroupEntry {
//                    binding: 2,
//                    resource: output_vector_buffer.as_entire_binding(),
//                },
//            ],
//        });
//
//        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//            label: None,
//            bind_group_layouts: &[&bind_group_layout],
//            push_constant_ranges: &[],
//        });
//        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
//            label: None,
//            layout: Some(&pipeline_layout),
//            module: &shader,
//            entry_point: "main",
//            compilation_options: Default::default(),
//            cache: None,
//        });
//
//        WgpuContext {
//            device,
//            queue,
//            pipeline,
//            bind_group,
//            output_staging_buffer,
//            input_vector_buffer,
//            matrix_buffer,
//            output_vector_buffer,
//        }
//    }
//}

enum TensorContent {
    InGPUMemory,
    InRam(Vec<u32>),
}

/// the type for KV-cache et value-cache
pub struct Cache<'a> {
    gpu_context: &'a WgpuContext<'a>,
}

impl<'a> Cache<'a> {
    pub fn new(gpu_context: &'a Option<WgpuContext<'a>>, gpu_index: GpuIndex) -> Cache<'a> {
        // les vecteurs sont initializés à 0.0
        todo!()
    }
}

impl<'a> Index<Range<usize>> for Cache<'a> {
    type Output = DropGuard<'a>;

    fn index(&self, range: Range<usize>) -> &Self::Output {
        todo!()
    }
}

impl<'a> IndexMut<Range<usize>> for Cache<'a> {
    //    type Output = DropGuard<'a>;

    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        todo!()
    }
}

pub struct DropGuard<'a>(Vector<'a>);

//The dropguard is there to make sure that the vectors taken from the kv and value caches are updated on the GPU side when being dropped
impl<'a> Drop for DropGuard<'a> {
    fn drop(&mut self) {
        todo!()
    }
}

impl<'a> Deref for DropGuard<'a> {
    type Target = Vector<'a>;

    // Required method
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for DropGuard<'a> {
    // Required method
    fn deref_mut(&mut self) -> &mut Vector<'a> {
        &mut self.0
    }
}

pub struct Vector<'a> {
    gpu_context: &'a WgpuContext<'a>,
    range_start: u32,
    range_end: u32,
    tensor_content: TensorContent,
}

impl<'a> Vector<'a> {
    pub fn new(gpu_context: &'a Option<WgpuContext<'a>>, gpu_index: GpuIndex) -> Vector<'a> {
        // les vecteurs sont initializés à 0.0
        todo!()
    }

    /// Get immutable access to a slice of the data in the tensor
    /// If the data is on the GPU, fetch it first
    pub fn data(&self) -> &[f32] {
        todo!()
    }

    /// Get immutable access to a slice of the data in the tensor
    /// If the data is on the GPU, fetch it first
    pub fn data_mut(&mut self) -> &mut [f32] {
        todo!()
    }
}

pub struct Matrix<'a> {
    gpu_context: &'a WgpuContext<'a>,
    range_start: u32,
    range_end: u32,
    tensor_content: TensorContent,
}

impl<'a> Matrix<'a> {
    pub fn new(gpu_context: &'a WgpuContext, range_start: u32, range_end: u32) -> Matrix<'a> {
        todo!()
    }

    /// Get immutable access to a slice of the data in the tensor
    /// Since the data is immutable, we get it directly from the original data located on the CPU side, we don't have to load it from the GPU
    pub fn data(&self) -> &[f32] {
        todo!()
    }
}

/// Weights are bags of matrices
pub struct Weights<'a> {
    gpu_context: &'a WgpuContext<'a>,
}

impl<'a> Weights<'a> {
    pub fn new(gpu_context: &'a WgpuContext<'a>, begining: u32, end: u32) -> Weights<'a> {
        todo!()
    }

    pub fn as_matrix(&self) -> &Matrix<'a> {
        todo!()
    }
}

/// Since Weight are immutable, we can do a shallow clone
impl<'a> Clone for Weights<'a> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'a> Index<Range<usize>> for Weights<'a> {
    type Output = Matrix<'a>;

    fn index(&self, range: Range<usize>) -> &Self::Output {
        todo!()
    }
}

pub fn matmul<'a>(output: &mut Vector<'a>, input: &Vector<'a>, matrix: &Matrix<'a>) {
    assert!(
        std::ptr::eq(input.gpu_context, matrix.gpu_context),
        "Input vector and matrix must live in the same GPU context"
    );
    todo!()
}
