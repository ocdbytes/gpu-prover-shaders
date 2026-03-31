use ark_bn254::Fr as Bn254Field;
use ark_ff::PrimeField;

use super::Bn254;

pub trait Conversion {
    fn to_limbs(&self) -> [u32; 8];
    fn from_limbs(limbs: [u32; 8]) -> Self;
    /// Raw Montgomery-form limbs (no conversion out of Montgomery space).
    /// Use this when sending data to the GPU shader which operates in Montgomery form.
    fn to_mont_limbs(&self) -> [u32; 8];
    /// Construct from raw Montgomery-form limbs (no conversion into Montgomery space).
    fn from_mont_limbs(limbs: [u32; 8]) -> Self;
}

impl Conversion for Bn254 {
    fn to_limbs(&self) -> [u32; 8] {
        let mut limbs = [0u32; 8];
        let value = self.into_bigint().0; // [u64; 4]
        for i in 0..4 {
            limbs[2 * i] = value[i] as u32;
            limbs[2 * i + 1] = (value[i] >> 32) as u32;
        }
        limbs
    }

    fn from_limbs(limbs: [u32; 8]) -> Self {
        let mut value = <Bn254Field as PrimeField>::BigInt::zero();
        for i in 0..4 {
            value.0[i] = limbs[2 * i] as u64 | ((limbs[2 * i + 1] as u64) << 32);
        }
        Bn254::new(value)
    }

    fn to_mont_limbs(&self) -> [u32; 8] {
        let mut limbs = [0u32; 8];
        let value = self.0 .0; // raw [u64; 4] in Montgomery form
        for i in 0..4 {
            limbs[2 * i] = value[i] as u32;
            limbs[2 * i + 1] = (value[i] >> 32) as u32;
        }
        limbs
    }

    fn from_mont_limbs(limbs: [u32; 8]) -> Self {
        let mut value = <Bn254Field as PrimeField>::BigInt::zero();
        for i in 0..4 {
            value.0[i] = limbs[2 * i] as u64 | ((limbs[2 * i + 1] as u64) << 32);
        }
        Bn254Field::new_unchecked(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::constants::*;
    use ark_bn254::FrConfig;
    use ark_ff::fields::models::fp::MontConfig;

    #[test]
    fn test_p_limbs() {
        let m = FrConfig::MODULUS.0;
        for i in 0..4 {
            let lo = P_LIMBS[2 * i] as u64;
            let hi = (P_LIMBS[2 * i + 1] as u64) << 32;
            assert_eq!(lo | hi, m[i], "P_LIMBS mismatch at u64 limb {i}");
        }
    }

    #[test]
    fn test_r_squared() {
        let r2 = FrConfig::R2.0;
        for i in 0..4 {
            let lo = R_SQUARED[2 * i] as u64;
            let hi = (R_SQUARED[2 * i + 1] as u64) << 32;
            assert_eq!(lo | hi, r2[i], "R_SQUARED mismatch at u64 limb {i}");
        }
    }

    #[test]
    fn test_p_inv() {
        let p0 = P_LIMBS[0] as u64;
        let inv = P_INV as u64;
        let product = (p0 * inv) as u32;
        assert_eq!(product, u32::MAX, "p₀ * P_INV should ≡ -1 (mod 2³²)");
    }

    #[test]
    fn test_conversion_roundtrip() {
        use ark_std::UniformRand;
        let mut rng = ark_std::test_rng();
        for _ in 0..100 {
            let val = Bn254::rand(&mut rng);
            let limbs = val.to_limbs();
            let recovered = Bn254::from_limbs(limbs);
            assert_eq!(val, recovered);
        }
    }
}
