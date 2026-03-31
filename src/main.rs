use metal_learn::bench;
use metal_learn::gpu::GpuContext;

fn main() {
    let gpu = GpuContext::new();

    bench::bench_mul(&gpu);
    bench::bench_scalar_mul_add(&gpu);
    bench::bench_eval_eq(&gpu);
    bench::bench_fold(&gpu);
}
