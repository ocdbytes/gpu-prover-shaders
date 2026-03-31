use metal::*;
use std::collections::HashMap;
use std::ffi::c_void;

const LIB_FIELD_DATA: &[u8] = include_bytes!("../../shaders/dotprod_field.metallib");
const LIB_EVAL_EQ_DATA: &[u8] = include_bytes!("../../shaders/eval_eq.metallib");
const LIB_FOLD_DATA: &[u8] = include_bytes!("../../shaders/fold.metallib");

const BUFFER_OPTIONS: MTLResourceOptions = MTLResourceOptions::from_bits_truncate(
    MTLResourceOptions::StorageModeShared.bits()
        | MTLResourceOptions::HazardTrackingModeUntracked.bits(),
);

pub struct GpuContext {
    device: Device,
    pipelines: HashMap<&'static str, ComputePipelineState>,
    command_queue: CommandQueue,
}

impl GpuContext {
    pub fn new() -> Self {
        let device = Device::system_default().expect("No device found");
        println!("Device: {:?}", device);
        let command_queue = device.new_command_queue();

        let mut pipelines = HashMap::new();

        // Load field kernels
        let field_lib = device
            .new_library_with_data(LIB_FIELD_DATA)
            .expect("Failed to load field metal lib");
        for name in ["dot_product_field", "scalar_mul_add_field"] {
            let pipeline = Self::make_pipeline(&device, &field_lib, name);
            pipelines.insert(name, pipeline);
        }

        // Load eval_eq kernel
        let eval_eq_lib = device
            .new_library_with_data(LIB_EVAL_EQ_DATA)
            .expect("Failed to load eval_eq metal lib");
        pipelines.insert("eval_eq_field", Self::make_pipeline(&device, &eval_eq_lib, "eval_eq_field"));

        // Load fold kernel
        let fold_lib = device
            .new_library_with_data(LIB_FOLD_DATA)
            .expect("Failed to load fold metal lib");
        pipelines.insert("fold_field", Self::make_pipeline(&device, &fold_lib, "fold_field"));

        Self {
            device,
            pipelines,
            command_queue,
        }
    }

    fn make_pipeline(device: &Device, lib: &Library, name: &str) -> ComputePipelineState {
        let function = lib
            .get_function(name, None)
            .unwrap_or_else(|_| panic!("Failed to find {name} kernel"));
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("Failed to create pipeline")
    }

    pub fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.pipelines
            .get(name)
            .unwrap_or_else(|| panic!("Pipeline '{name}' not found"))
    }

    // ── Buffer creation ──────────────────────────────────────────

    /// Zero-copy buffer wrapping existing page-aligned memory.
    /// The caller MUST keep the data alive for the lifetime of the returned Buffer.
    pub unsafe fn wrap_buffer_no_copy(&self, data: &[u32]) -> Buffer {
        let size = (data.len() * std::mem::size_of::<u32>()) as u64;
        self.device.new_buffer_with_bytes_no_copy(
            data.as_ptr() as *const c_void,
            size,
            BUFFER_OPTIONS,
            None,
        )
    }

    /// Allocating buffer that copies data in.
    pub fn create_buffer(&self, data: &[u32]) -> Buffer {
        let size = (data.len() * std::mem::size_of::<u32>()) as u64;
        self.device
            .new_buffer_with_data(data.as_ptr() as *const c_void, size, BUFFER_OPTIONS)
    }

    /// Allocate a zeroed buffer.
    pub fn create_buffer_zeroed(&self, num_u32s: usize) -> Buffer {
        let size = (num_u32s * std::mem::size_of::<u32>()) as u64;
        self.device.new_buffer(size, BUFFER_OPTIONS)
    }

    // ── Dispatch ─────────────────────────────────────────────────

    /// Single dispatch: encode one kernel, commit, and wait.
    pub fn dispatch(&self, pipeline_name: &str, buffers: &[&Buffer], num_threads: u64) {
        let pipeline = self.pipeline(pipeline_name);
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encode_dispatch(encoder, pipeline, buffers, num_threads);
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Dispatch with inline bytes arguments (for kernels that take scalar params).
    ///
    /// `buffers` are bound starting at index 0.
    /// `bytes_args` are bound at indices starting after the last buffer.
    pub fn dispatch_with_bytes(
        &self,
        pipeline_name: &str,
        buffers: &[&Buffer],
        bytes_args: &[&[u8]],
        num_threads: u64,
    ) {
        let pipeline = self.pipeline(pipeline_name);
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        for (i, buf) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buf), 0);
        }
        for (i, bytes) in bytes_args.iter().enumerate() {
            encoder.set_bytes(
                (buffers.len() + i) as u64,
                bytes.len() as u64,
                bytes.as_ptr() as *const _,
            );
        }

        let grid_size = MTLSize::new(num_threads, 1, 1);
        let tg_size = threadgroup_size_for(pipeline, num_threads);
        encoder.dispatch_threads(grid_size, tg_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Batch dispatch: encode multiple kernel invocations into one command buffer.
    pub fn dispatch_batch(&self, dispatches: &[(&str, &[&Buffer], u64)]) {
        let command_buffer = self.command_queue.new_command_buffer();
        for &(pipeline_name, buffers, num_threads) in dispatches {
            let pipeline = self.pipeline(pipeline_name);
            let encoder = command_buffer.new_compute_command_encoder();
            encode_dispatch(encoder, pipeline, buffers, num_threads);
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

/// Compute threadgroup size aligned to the pipeline's SIMD width.
///
/// Apple Silicon executes threads in SIMD groups (typically 32 threads).
/// Aligning the threadgroup size to this width avoids partial SIMD groups
/// and improves scheduling efficiency for small dispatches.
fn threadgroup_size_for(pipeline: &ComputePipelineState, num_threads: u64) -> MTLSize {
    let simd_width = pipeline.thread_execution_width();
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    // Clamp to grid size, round up to SIMD width, cap at pipeline max
    let clamped = num_threads.min(max_threads);
    let aligned = ((clamped + simd_width - 1) / simd_width) * simd_width;
    MTLSize::new(aligned.min(max_threads), 1, 1)
}

fn encode_dispatch(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    num_threads: u64,
) {
    encoder.set_compute_pipeline_state(pipeline);
    let buf_refs: Vec<Option<&BufferRef>> = buffers.iter().map(|b| Some(b.as_ref())).collect();
    let offsets = vec![0u64; buffers.len()];
    encoder.set_buffers(0, &buf_refs, &offsets);

    let grid_size = MTLSize::new(num_threads, 1, 1);
    let tg_size = threadgroup_size_for(pipeline, num_threads);
    encoder.dispatch_threads(grid_size, tg_size);
    encoder.end_encoding();
}
