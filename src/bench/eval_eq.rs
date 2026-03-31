use std::time::Instant;

use ark_bn254::Fr;
use ark_std::UniformRand;

use crate::field::Conversion;
use crate::gpu::GpuContext;

use super::verify_results;

/// CPU eval_eq using whir's actual algorithm: recursive tree with rayon parallelism.
fn cpu_eval_eq_recursive(accumulator: &mut [Fr], point: &[Fr], scalar: Fr) {
    assert_eq!(accumulator.len(), 1 << point.len());
    if let Some((&x0, rest)) = point.split_first() {
        let half = accumulator.len() / 2;
        let (lo, hi) = accumulator.split_at_mut(half);
        let s1 = scalar * x0;
        let s0 = scalar - s1;
        if lo.len() > 4096 {
            rayon::join(
                || cpu_eval_eq_recursive(lo, rest, s0),
                || cpu_eval_eq_recursive(hi, rest, s1),
            );
        } else {
            cpu_eval_eq_recursive(lo, rest, s0);
            cpu_eval_eq_recursive(hi, rest, s1);
        }
    } else {
        accumulator[0] += scalar;
    }
}

/// Naive per-element CPU eval_eq (for verification only).
fn cpu_eval_eq_naive(accumulator: &mut [Fr], point: &[Fr], scalar: Fr) {
    let num_vars = point.len();
    let one = Fr::from(1u64);
    for i in 0..accumulator.len() {
        let mut prod = scalar;
        for (k, &pk) in point.iter().enumerate() {
            let bit = (i >> (num_vars - 1 - k)) & 1;
            if bit == 1 {
                prod *= pk;
            } else {
                prod *= one - pk;
            }
        }
        accumulator[i] += prod;
    }
}

pub fn bench_eval_eq(gpu: &GpuContext) {
    let test_num_vars: &[u32] = &[12, 16, 18, 20, 22];
    let iters = 5;

    println!("\n=== Eval Eq Sweep ===");

    for &num_vars in test_num_vars {
        let size = 1usize << num_vars;
        let mut rng = ark_std::test_rng();

        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let scalar = Fr::rand(&mut rng);

        // CPU (whir's recursive + rayon)
        let mut cpu_times = Vec::with_capacity(iters);
        let mut cpu_acc = vec![Fr::from(0u64); size];
        for _ in 0..iters {
            cpu_acc.fill(Fr::from(0u64));
            let start = Instant::now();
            cpu_eval_eq_recursive(&mut cpu_acc, &point, scalar);
            cpu_times.push(start.elapsed());
        }
        cpu_times.sort();
        let cpu_median = cpu_times[iters / 2];

        // GPU
        let point_limbs: Vec<u32> = point.iter().flat_map(|f| f.to_mont_limbs()).collect();
        let scalar_limbs: Vec<u32> = scalar.to_mont_limbs().to_vec();

        let buf_acc = gpu.create_buffer_zeroed(size * 8);
        let buf_point = gpu.create_buffer(&point_limbs);
        let buf_scalar = gpu.create_buffer(&scalar_limbs);
        let num_vars_bytes = num_vars.to_ne_bytes();

        // Warmup
        gpu.dispatch_with_bytes(
            "eval_eq_field",
            &[&buf_acc, &buf_point, &buf_scalar],
            &[&num_vars_bytes],
            1u64 << num_vars,
        );

        let mut gpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            unsafe {
                std::ptr::write_bytes(buf_acc.contents() as *mut u8, 0, size * 8 * 4);
            }
            let start = Instant::now();
            gpu.dispatch_with_bytes(
                "eval_eq_field",
                &[&buf_acc, &buf_point, &buf_scalar],
                &[&num_vars_bytes],
                1u64 << num_vars,
            );
            gpu_times.push(start.elapsed());
        }
        gpu_times.sort();
        let gpu_median = gpu_times[iters / 2];

        // Verify at smallest sizes only (naive is too slow for large)
        if num_vars <= 16 {
            let mut verify_acc = vec![Fr::from(0u64); size];
            cpu_eval_eq_naive(&mut verify_acc, &point, scalar);

            let ptr = buf_acc.contents() as *const u32;
            let result_u32s = unsafe { std::slice::from_raw_parts(ptr, size * 8) };
            verify_results(result_u32s, &verify_acc, size, &format!("eval_eq d={num_vars}"));
        }

        let speedup = cpu_median.as_secs_f64() / gpu_median.as_secs_f64();
        println!(
            "  d={num_vars:2} | 2^{num_vars} = {:>8} elements | CPU: {:>10.2?} | GPU: {:>10.2?} | {:.2}x",
            size, cpu_median, gpu_median, speedup
        );
    }
}
