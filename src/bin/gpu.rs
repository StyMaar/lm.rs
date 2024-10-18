//! See hello-compute example main.rs for more details
//! as similar items here are not explained.
//!
//! This example does elaborate on some things though that the
//! hello-compute example does not such as mapping buffers
//! and why use the async channels.

use std::{marker::PhantomData, mem::size_of_val};
use wgpu::util::DeviceExt;

const OVERFLOW: u32 = 0xffffffff;

async fn run() {
    let matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1u32];

    let context = WgpuContext::new(4, &matrix).await;

    let vector = [[1, 0, 0, 0], [2, 2, 2, 2], [1, 2, 3, 4], [1, 4, 9, 16]];

    let mut output_vec = [0, 0, 0];
    for ve in &vector {
        compute_malmul(ve, &mut output_vec, &context).await;
        println!("Output of {ve:#?} x {matrix:#?} = {output_vec:#?}");
    }
}

// enum MatMulIO<'a> {
//     CpuInBuffer(&'a [u32]),
//     CpuOutBuffer(&'a mut [u32]),
//     GpuBuffer(&'a wgpu::Buffer),
// }

async fn compute_malmul(input_vec: &[u32], output_vec: &mut [u32], context: &WgpuContext) {
    log::info!("Beginning GPU compute on data {input_vec:?}.");
    // Local buffer contents -> GPU storage buffer
    // Adds a write buffer command to the queue. This command is more complicated
    // than it appears.
    context.queue.write_buffer(
        &context.input_vector_buffer,
        0,
        bytemuck::cast_slice(input_vec),
    );
    log::info!("Wrote to buffer.");

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
        compute_pass.dispatch_workgroups(output_vec.len() as u32, 1, 1);
    }
    // We finish the compute pass by dropping it.

    // Entire storage buffer -> staging buffer.
    command_encoder.copy_buffer_to_buffer(
        &context.output_vector_buffer,
        0,
        &context.output_staging_buffer,
        0,
        context.output_vector_buffer.size(),
    );

    // Finalize the command encoder, add the contained commands to the queue and flush.
    context.queue.submit(Some(command_encoder.finish()));
    log::info!("Submitted commands.");

    // Finally time to get our results.
    // First we get a buffer slice which represents a chunk of the buffer (which we
    // can't access yet).
    // We want the whole thing so use unbounded range.
    let buffer_slice = context.output_staging_buffer.slice(..);
    // Now things get complicated. WebGPU, for safety reasons, only allows either the GPU
    // or CPU to access a buffer's contents at a time. We need to "map" the buffer which means
    // flipping ownership of the buffer over to the CPU and making access legal. We do this
    // with `BufferSlice::map_async`.
    //
    // The problem is that map_async is not an async function so we can't await it. What
    // we need to do instead is pass in a closure that will be executed when the slice is
    // either mapped or the mapping has failed.
    //
    // The problem with this is that we don't have a reliable way to wait in the main
    // code for the buffer to be mapped and even worse, calling get_mapped_range or
    // get_mapped_range_mut prematurely will cause a panic, not return an error.
    //
    // Using channels solves this as awaiting the receiving of a message from
    // the passed closure will force the outside code to wait. It also doesn't hurt
    // if the closure finishes before the outside code catches up as the message is
    // buffered and receiving will just pick that up.
    //
    // It may also be worth noting that although on native, the usage of asynchronous
    // channels is wholly unnecessary, for the sake of portability to WASM (std channels
    // don't work on WASM,) we'll use async channels that work on both native and WASM.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    // In order for the mapping to be completed, one of three things must happen.
    // One of those can be calling `Device::poll`. This isn't necessary on the web as devices
    // are polled automatically but natively, we need to make sure this happens manually.
    // `Maintain::Wait` will cause the thread to wait on native but not on WebGpu.
    context
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();
    log::info!("Device polled.");
    // Now we await the receiving and panic if anything went wrong because we're lazy.
    receiver.recv_async().await.unwrap().unwrap();
    log::info!("Result received.");
    // NOW we can call get_mapped_range.
    {
        let view = buffer_slice.get_mapped_range();
        output_vec.copy_from_slice(bytemuck::cast_slice(&view));
    }
    log::info!("Results written to local buffer.");
    // We need to make sure all `BufferView`'s are dropped before we do what we're about
    // to do.
    // Unmap so that we can copy to the staging buffer in the next iteration.
    context.output_staging_buffer.unmap();
}

pub fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");

        crate::utils::add_web_nothing_to_see_msg();

        wasm_bindgen_futures::spawn_local(run());
    }
}

/// A convenient way to hold together all the useful wgpu stuff together.
struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    input_vector_buffer: wgpu::Buffer,
    matrix_buffer: wgpu::Buffer,
    output_vector_buffer: wgpu::Buffer,
    output_staging_buffer: wgpu::Buffer,
}

impl WgpuContext {
    async fn new(input_size: usize, matrix: &[u32]) -> WgpuContext {
        let output_size = matrix.len() / input_size; // matrix size being equal to input_size x output_size

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
        let shader = device.create_shader_module(wgpu::include_wgsl!("gpu/shader.wgsl"));

        let input_vector_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector Buffer"),
            mapped_at_creation: false,
            size: (input_size * 4usize) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix Buffer"),
            contents: bytemuck::cast_slice(&matrix),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let output_vector_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector Buffer"),
            size: (output_size * 4usize) as wgpu::BufferAddress,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (output_size * 4usize) as wgpu::BufferAddress,
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
                    resource: input_vector_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_vector_buffer.as_entire_binding(),
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
            output_staging_buffer,
            input_vector_buffer,
            matrix_buffer,
            output_vector_buffer,
        }
    }
}

