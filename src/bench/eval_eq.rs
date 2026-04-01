use std::time::Instant;

use ark_bn254::Fr;
use ark_std::UniformRand;

use crate::field::Conversion;
use crate::gpu::{GpuContext, GpuTiming};

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

/// Cutoff: the first CUTOFF levels are computed on CPU, remaining on GPU.
/// 2^10 = 1024 elements — instant on CPU, avoids tiny GPU dispatches.
const TREE_CUTOFF: usize = 10;

pub fn bench_eval_eq(gpu: &GpuContext) {
    let test_num_vars: &[u32] = &[11, 12, 14, 16, 18, 19, 20, 22];
    let iters = 5;

    println!("\n=== Eval Eq Sweep (Tree) ===");

    for &num_vars in test_num_vars {
        let n = num_vars as usize;
        let size = 1usize << n;
        let mut rng = ark_std::test_rng();

        let point: Vec<Fr> = (0..num_vars).map(|_| Fr::rand(&mut rng)).collect();
        let scalar = Fr::rand(&mut rng);

        // ── CPU benchmark (recursive + rayon) ────────────────────────
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

        // ── GPU tree benchmark ───────────────────────────────────────
        let cutoff = TREE_CUTOFF.min(n);

        // CPU seed: compute the first `cutoff` levels on CPU
        let cpu_seed_size = 1usize << cutoff;
        let mut seed = vec![Fr::from(0u64); cpu_seed_size];
        cpu_eval_eq_recursive(&mut seed, &point[..cutoff], scalar);
        let seed_limbs: Vec<u32> = seed.iter().flat_map(|f| f.to_mont_limbs()).collect();

        // GPU buffers (double-buffered ping-pong)
        let point_limbs: Vec<u32> = point.iter().flat_map(|f| f.to_mont_limbs()).collect();
        let buf_a = gpu.create_buffer_zeroed(size * 8);
        let buf_b = gpu.create_buffer_zeroed(size * 8);
        let buf_point = gpu.create_buffer(&point_limbs);

        // Upload seed into buf_a
        unsafe {
            std::ptr::copy_nonoverlapping(
                seed_limbs.as_ptr(),
                buf_a.contents() as *mut u32,
                seed_limbs.len(),
            );
        }

        // Warmup
        gpu.dispatch_eval_eq_tree(&buf_a, &buf_b, &buf_point, cutoff..n);

        // Timed runs
        let mut gpu_timings: Vec<GpuTiming> = Vec::with_capacity(iters);
        let mut result_in_b = false;
        for _ in 0..iters {
            // Reset buf_a with seed data
            unsafe {
                std::ptr::copy_nonoverlapping(
                    seed_limbs.as_ptr(),
                    buf_a.contents() as *mut u32,
                    seed_limbs.len(),
                );
            }
            let (rib, t) = gpu.dispatch_eval_eq_tree_timed(&buf_a, &buf_b, &buf_point, cutoff..n);
            result_in_b = rib;
            gpu_timings.push(t);
        }
        gpu_timings.sort_by_key(|t| t.total);
        let med = gpu_timings[iters / 2];

        // Verify at smallest sizes (naive is O(n * 2^n), too slow for large n)
        if num_vars <= 16 {
            let result_buf = if result_in_b { &buf_b } else { &buf_a };
            let mut verify_acc = vec![Fr::from(0u64); size];
            cpu_eval_eq_naive(&mut verify_acc, &point, scalar);

            let ptr = result_buf.contents() as *const u32;
            let result_u32s = unsafe { std::slice::from_raw_parts(ptr, size * 8) };
            verify_results(
                result_u32s,
                &verify_acc,
                size,
                &format!("eval_eq_tree d={num_vars}"),
            );
        }

        let speedup = cpu_median.as_secs_f64() / med.total.as_secs_f64();
        println!(
            "  2^{num_vars:2} = {size:>8} | CPU: {:>10.2?} | GPU: {:>10.2?} (compute: {:>10.2?}) | {speedup:.2}x",
            cpu_median, med.total, med.compute
        );
    }
}
