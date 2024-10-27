use std::clone::Clone;
use std::cmp;
use std::ops::{Deref, DerefMut, Drop, Index, IndexMut, Range};
use wgpu::util::DeviceExt;

pub struct WgpuContextBuilder {
    state_buffer_size: usize,
    output_staging_buffer_size: usize,
}

impl WgpuContextBuilder {
    pub fn new() -> WgpuContextBuilder {
        WgpuContextBuilder {
            //         data,
            state_buffer_size: 0,
            output_staging_buffer_size: 0,
        }
    }

    pub fn make_state(&mut self, size: usize) -> GpuIndex {
        let index_begin = self.state_buffer_size;
        let index_end = self.state_buffer_size + size;

        self.state_buffer_size += size;
        self.output_staging_buffer_size = cmp::max(self.output_staging_buffer_size, size);

        GpuIndex {
            begin: index_begin,
            end: index_end,
        }
    }

    // create the GPU context
    pub async fn finalize<'a>(self, data: &'a [u8]) -> WgpuContext<'a> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        // Our shader, kindly compiled with Naga.
        let shader = device.create_shader_module(wgpu::include_wgsl!("matmul.wgsl"));

        let WgpuContextBuilder {
            state_buffer_size,
            output_staging_buffer_size,
        } = self;

        let state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("State Buffer"),
            contents: bytemuck::cast_slice(&vec!), // TODO is bytemuck necessary at all?
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC, //COPY_DST et COPY_SRC ne sont sans doute pas nécessaire pour ce buffer
        });

        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Weight Buffer"),
            contents: bytemuck::cast_slice(&data), // TODO is bytemuck necessary at all?
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC, //COPY_DST et COPY_SRC ne sont sans doute pas nécessaire pour ce buffer
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Index Buffer"),
            size: 16 as wgpu::BufferAddress, // 4 u32 indexes, 4x4bytes = 16 bytes
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC, //COPY_SRC est sans doute inutile
        });

        let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (output_staging_buffer_size) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // This can be though of as the function signature for our CPU-GPU function.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        // Going to have this be None just to be safe.
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        // Going to have this be None just to be safe.
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        // Going to have this be None just to be safe.
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        // This ties actual resources stored in the GPU to our metaphorical function
        // through the binding slots we defined above.

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: index_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        WgpuContext {
            device,
            queue,
            pipeline,
            bind_group,
            state_buffer,
            weights_buffer,
            index_buffer,
            output_staging_buffer,
            data,
        }
    }
}

pub struct GpuIndex {
    begin: usize,
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
    index_buffer: wgpu::Buffer, // buffer where the indexes of the multiplication parameters are stored
    output_staging_buffer: wgpu::Buffer, // the buffer to send value back to the CPU
}

enum TensorContent {
    InGPUMemory,
    InRam(Vec<u32>),
}

/// the type for KV-cache et value-cache
pub struct Cache<'a> {
    gpu_context: &'a WgpuContext<'a>,
}

impl<'a> Cache<'a> {
    pub fn new(gpu_context: &'a WgpuContext<'a>, gpu_index: GpuIndex) -> Cache<'a> {
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
    pub fn new(gpu_context: &'a WgpuContext<'a>, gpu_index: GpuIndex) -> Vector<'a> {
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
    begining: u32,
    end: u32,
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

// La multiplication a lieu entre le statebuffer (qui contient l'input et l'output) et le weightbuffer (qui contient la matrice)
// la destination de la multiplication est le temporary buffer et on doit ensuite copier le résultat vers le statebuffer
// remarque en fait on va plutôt faire directement la multiplication au bon endroit dans le staging buffer
//
pub fn matmul<'a>(output: &mut Vector<'a>, input: &Vector<'a>, matrix: &Matrix<'a>) {
    assert!(
        std::ptr::eq(input.gpu_context, matrix.gpu_context),
        "Input vector and matrix must live in the same GPU context"
    ); // TODO vérifier aussi pour output

    let context = input.gpu_context;

    // if the Vector content has been copied to CPU memory to be mutated, copy it back to the GPU memory before doing the operation
    if let TensorContent::InRam(ref input_vec) = input.tensor_content {
        // Local buffer contents -> GPU storage buffer
        // Adds a write buffer command to the queue. This command is more complicated
        // than it appears.

        context.queue.write_buffer(
            &context.state_buffer,
            input.range_start as u64,        // TODO: check off by one error
            bytemuck::cast_slice(input_vec), //TODO je ne suis pas sûr que bytemuck serve à quelque chose ici
        );
        log::info!("Wrote to state buffer.");
    }

    
        context.queue.write_buffer(
            &context.index_buffer,
            0u64,
            bytemuck::cast_slice(&[matrix.range_start, input.range_start, output.range_start, output.range_end]),
        );

    let mut command_encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&context.pipeline);
        compute_pass.set_bind_group(0, &context.bind_group, &[]);

        let output_size = output.range_end - output.range_start + 1; // TODO check off by one error
        compute_pass.dispatch_workgroups(output_size as u32, 1, 1);
    }
    // We finish the compute pass by dropping it.

    // Finalize the command encoder, add the contained commands to the queue and flush.
    context.queue.submit(Some(command_encoder.finish()));
    log::info!("Submitted commands.");

    // TODO il est possible qu'on ait besoin de faire un device.poll ici.
}

//    // Finally time to get our results.
//    // First we get a buffer slice which represents a chunk of the buffer (which we
//    // can't access yet).
//    // We want the whole thing so use unbounded range.
//    let buffer_slice = context.output_staging_buffer.slice(..);
//    // Now things get complicated. WebGPU, for safety reasons, only allows either the GPU
//    // or CPU to access a buffer's contents at a time. We need to "map" the buffer which means
//    // flipping ownership of the buffer over to the CPU and making access legal. We do this
//    // with `BufferSlice::map_async`.
//    //
//    // The problem is that map_async is not an async function so we can't await it. What
//    // we need to do instead is pass in a closure that will be executed when the slice is
//    // either mapped or the mapping has failed.
//    //
//    // The problem with this is that we don't have a reliable way to wait in the main
//    // code for the buffer to be mapped and even worse, calling get_mapped_range or
//    // get_mapped_range_mut prematurely will cause a panic, not return an error.
//    //
//    // Using channels solves this as awaiting the receiving of a message from
//    // the passed closure will force the outside code to wait. It also doesn't hurt
//    // if the closure finishes before the outside code catches up as the message is
//    // buffered and receiving will just pick that up.
//
//    (sender, receiver) = std::sync::mpsc();
//    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
//    // In order for the mapping to be completed, one of three things must happen.
//    // One of those can be calling `Device::poll`. This isn't necessary on the web as devices
//    // are polled automatically but natively, we need to make sure this happens manually.
//    // `Maintain::Wait` will cause the thread to wait on native but not on WebGpu.
//    context
//        .device
//        .poll(wgpu::Maintain::wait())
//        .panic_on_timeout();
//    log::info!("Device polled.");
//    // Now we await the receiving and panic if anything went wrong because we're lazy.
//    receiver.recv().unwrap().unwrap();
//    log::info!("Result received.");
//    // NOW we can call get_mapped_range.
//    {
//        let view = buffer_slice.get_mapped_range();
//        output_vec.copy_from_slice(bytemuck::cast_slice(&view));
//    }
//    log::info!("Results written to local buffer.");
//    // We need to make sure all `BufferView`'s are dropped before we do what we're about
//    // to do.
//    // Unmap so that we can copy to the staging buffer in the next iteration.
//    context.output_staging_buffer.unmap();
//
