use std::time::Instant;

use ark_bn254::Fr;
use ark_std::UniformRand;
use rayon::prelude::*;

use crate::field::Conversion;
use crate::gpu::{GpuContext, GpuTiming, to_page_aligned};

use super::verify_results;

pub fn bench_mul(gpu: &GpuContext) {
    let test_sizes: &[usize] = &[
        1 << 11, // 2048
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 19,
        1 << 20,
        1 << 22,
    ];
    let iters = 5;

    println!("\n=== Element-wise Multiply Sweep ===");

    for &n in test_sizes {
        let mut rng = ark_std::test_rng();
        let a_field: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let b_field: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        // CPU (parallel)
        let mut cpu_times = Vec::with_capacity(iters);
        let mut cpu_results_buf: Vec<Fr> = vec![Fr::from(0u64); n];
        for _ in 0..iters {
            let start = Instant::now();
            cpu_results_buf
                .par_iter_mut()
                .zip(a_field.par_iter().zip(b_field.par_iter()))
                .for_each(|(out, (a, b))| *out = *a * *b);
            cpu_times.push(start.elapsed());
        }
        cpu_times.sort();
        let cpu_median = cpu_times[iters / 2];

        // GPU
        let a_limbs: Vec<u32> = a_field.iter().flat_map(|f| f.to_mont_limbs()).collect();
        let b_limbs: Vec<u32> = b_field.iter().flat_map(|f| f.to_mont_limbs()).collect();
        let a_aligned = to_page_aligned(&a_limbs);
        let b_aligned = to_page_aligned(&b_limbs);

        let buf_a = unsafe { gpu.wrap_buffer_no_copy(&a_aligned) };
        let buf_b = unsafe { gpu.wrap_buffer_no_copy(&b_aligned) };
        let buf_result = gpu.create_buffer_zeroed(n * 8);

        // Warmup
        gpu.dispatch("dot_product_field", &[&buf_a, &buf_b, &buf_result], n as u64);

        let mut gpu_timings: Vec<GpuTiming> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = gpu.dispatch_timed("dot_product_field", &[&buf_a, &buf_b, &buf_result], n as u64);
            gpu_timings.push(t);
        }
        gpu_timings.sort_by_key(|t| t.total);
        let med = gpu_timings[iters / 2];

        if n <= 65536 {
            let ptr = buf_result.contents() as *const u32;
            let result_u32s = unsafe { std::slice::from_raw_parts(ptr, n * 8) };
            verify_results(result_u32s, &cpu_results_buf, n, &format!("mul n={n}"));
        }

        let d = n.trailing_zeros();
        let speedup = cpu_median.as_secs_f64() / med.total.as_secs_f64();
        println!(
            "  2^{d:2} = {n:>8} | CPU: {:>10.2?} | GPU: {:>10.2?} (compute: {:>10.2?}) | {speedup:.2}x",
            cpu_median, med.total, med.compute
        );
    }
}

pub fn bench_scalar_mul_add(gpu: &GpuContext) {
    let test_sizes: &[usize] = &[
        1 << 11, // 2048
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 19,
        1 << 20,
        1 << 22,
    ];
    let iters = 5;

    println!("\n=== Scalar Mul-Add Sweep: acc[i] += weight * vec[i] ===");

    for &n in test_sizes {
        let mut rng = ark_std::test_rng();
        let weight = Fr::rand(&mut rng);
        let vector: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
        let accumulator: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();

        // CPU (parallel)
        let mut cpu_times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut acc = accumulator.clone();
            let start = Instant::now();
            acc.par_iter_mut()
                .zip(vector.par_iter())
                .for_each(|(a, v)| *a += weight * *v);
            cpu_times.push(start.elapsed());
        }
        cpu_times.sort();
        let cpu_median = cpu_times[iters / 2];

        // CPU reference for verification
        let cpu_results: Vec<Fr> = accumulator
            .iter()
            .zip(vector.iter())
            .map(|(acc, v)| *acc + weight * *v)
            .collect();

        // GPU
        let weight_limbs: Vec<u32> = weight.to_mont_limbs().to_vec();
        let vector_limbs: Vec<u32> = vector.iter().flat_map(|f| f.to_mont_limbs()).collect();
        let acc_limbs: Vec<u32> = accumulator.iter().flat_map(|f| f.to_mont_limbs()).collect();
        let vector_aligned = to_page_aligned(&vector_limbs);

        let buf_acc = gpu.create_buffer(&acc_limbs);
        let buf_weight = gpu.create_buffer(&weight_limbs);
        let buf_vector = unsafe { gpu.wrap_buffer_no_copy(&vector_aligned) };

        // Warmup
        gpu.dispatch("scalar_mul_add_field", &[&buf_acc, &buf_weight, &buf_vector], n as u64);

        let mut gpu_timings: Vec<GpuTiming> = Vec::with_capacity(iters);
        for _ in 0..iters {
            // Reset accumulator
            unsafe {
                std::ptr::copy_nonoverlapping(
                    acc_limbs.as_ptr(),
                    buf_acc.contents() as *mut u32,
                    acc_limbs.len(),
                );
            }
            let t = gpu.dispatch_timed("scalar_mul_add_field", &[&buf_acc, &buf_weight, &buf_vector], n as u64);
            gpu_timings.push(t);
        }
        gpu_timings.sort_by_key(|t| t.total);
        let med = gpu_timings[iters / 2];

        if n <= 65536 {
            let ptr = buf_acc.contents() as *const u32;
            let result_u32s = unsafe { std::slice::from_raw_parts(ptr, n * 8) };
            verify_results(result_u32s, &cpu_results, n, &format!("sma n={n}"));
        }

        let d = n.trailing_zeros();
        let speedup = cpu_median.as_secs_f64() / med.total.as_secs_f64();
        println!(
            "  2^{d:2} = {n:>8} | CPU: {:>10.2?} | GPU: {:>10.2?} (compute: {:>10.2?}) | {speedup:.2}x",
            cpu_median, med.total, med.compute
        );
    }
}
