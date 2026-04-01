use std::time::Instant;

use ark_bn254::Fr;
use ark_std::UniformRand;

use crate::field::Conversion;
use crate::gpu::{GpuContext, GpuTiming};

use super::verify_results;

/// CPU fold using whir's actual algorithm: recursive with rayon parallelism.
fn cpu_fold_recursive(values: &mut Vec<Fr>, weight: Fr) {
    fn recurse(low: &mut [Fr], high: &[Fr], weight: Fr) {
        if low.len() > 4096 {
            let split = low.len() / 2;
            let (ll, lr) = low.split_at_mut(split);
            let (hl, hr) = high.split_at(split);
            rayon::join(|| recurse(ll, hl, weight), || recurse(lr, hr, weight));
            return;
        }
        for (lo, hi) in low.iter_mut().zip(high) {
            *lo += (*hi - *lo) * weight;
        }
    }

    let half = values.len() / 2;
    let (low, high) = values.split_at_mut(half);
    recurse(low, high, weight);
    values.truncate(half);
}

pub fn bench_fold(gpu: &GpuContext) {
    let test_sizes: &[u32] = &[11, 12, 14, 16, 18, 19, 20, 22];
    let iters = 5;

    println!("\n=== Fold Sweep ===");

    for &num_vars in test_sizes {
        let full_size = 1usize << num_vars;
        let half = full_size / 2;
        let mut rng = ark_std::test_rng();

        let weight = Fr::rand(&mut rng);
        let original: Vec<Fr> = (0..full_size).map(|_| Fr::rand(&mut rng)).collect();

        // CPU (whir's recursive + rayon)
        let mut cpu_work = original.clone();
        let mut cpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            cpu_work.clear();
            cpu_work.extend_from_slice(&original);
            let start = Instant::now();
            cpu_fold_recursive(&mut cpu_work, weight);
            cpu_times.push(start.elapsed());
        }
        cpu_times.sort();
        let cpu_median = cpu_times[iters / 2];

        // Compute CPU reference result for verification
        let mut cpu_ref = original.clone();
        cpu_fold_recursive(&mut cpu_ref, weight);

        // GPU
        let original_limbs: Vec<u32> = original.iter().flat_map(|f| f.to_mont_limbs()).collect();
        let weight_limbs: Vec<u32> = weight.to_mont_limbs().to_vec();

        let buf_values = gpu.create_buffer(&original_limbs);
        let buf_weight = gpu.create_buffer(&weight_limbs);
        let half_size = half as u32;
        let half_size_bytes = half_size.to_ne_bytes();

        // Warmup
        gpu.dispatch_with_bytes(
            "fold_field",
            &[&buf_values, &buf_weight],
            &[&half_size_bytes],
            half as u64,
        );

        let mut gpu_timings: Vec<GpuTiming> = Vec::with_capacity(iters);
        for _ in 0..iters {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    original_limbs.as_ptr(),
                    buf_values.contents() as *mut u32,
                    original_limbs.len(),
                );
            }
            let t = gpu.dispatch_with_bytes_timed(
                "fold_field",
                &[&buf_values, &buf_weight],
                &[&half_size_bytes],
                half as u64,
            );
            gpu_timings.push(t);
        }
        gpu_timings.sort_by_key(|t| t.total);
        let med = gpu_timings[iters / 2];

        // Verify — only the first half_size elements matter
        let ptr = buf_values.contents() as *const u32;
        let result_u32s = unsafe { std::slice::from_raw_parts(ptr, half * 8) };
        verify_results(result_u32s, &cpu_ref, half, &format!("fold d={num_vars}"));

        let speedup = cpu_median.as_secs_f64() / med.total.as_secs_f64();
        println!(
            "  2^{num_vars:2} = {full_size:>8} | CPU: {:>10.2?} | GPU: {:>10.2?} (compute: {:>10.2?}) | {speedup:.2}x",
            cpu_median, med.total, med.compute
        );
    }
}
