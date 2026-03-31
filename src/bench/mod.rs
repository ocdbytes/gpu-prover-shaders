mod eval_eq;
mod fold;
mod mul;

pub use eval_eq::bench_eval_eq;
pub use fold::bench_fold;
pub use mul::{bench_mul, bench_scalar_mul_add};

use ark_bn254::Fr;

use crate::field::{Bn254, Conversion};

pub fn verify_results(gpu_u32s: &[u32], cpu_results: &[Fr], num: usize, label: &str) {
    let mut mismatches = 0;
    for i in 0..num {
        let offset = i * 8;
        let mut limbs = [0u32; 8];
        limbs.copy_from_slice(&gpu_u32s[offset..offset + 8]);
        let gpu_val = Bn254::from_mont_limbs(limbs);

        if gpu_val != cpu_results[i] {
            if mismatches < 5 {
                println!(
                    "[{}] MISMATCH at index {}: GPU={:?} CPU={:?}",
                    label, i, gpu_val, cpu_results[i]
                );
            }
            mismatches += 1;
        }
    }
    if mismatches == 0 {
        println!("[{}] All {} results match!", label, num);
    } else {
        println!("[{}] {}/{} mismatches", label, mismatches, num);
    }
}
