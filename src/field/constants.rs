use ark_bn254::FrConfig;
use ark_ff::fields::models::fp::MontConfig;
use ark_ff::PrimeField;

use super::Bn254;

pub const BN254_MOD: <Bn254 as PrimeField>::BigInt = Bn254::MODULUS;

/// BN254 scalar field modulus p as little-endian [u32; 8] limbs.
pub const P_LIMBS: [u32; 8] = {
    let m = FrConfig::MODULUS.0;
    [
        m[0] as u32,
        (m[0] >> 32) as u32,
        m[1] as u32,
        (m[1] >> 32) as u32,
        m[2] as u32,
        (m[2] >> 32) as u32,
        m[3] as u32,
        (m[3] >> 32) as u32,
    ]
};

/// R² mod p where R = 2^256, for converting into Montgomery form.
pub const R_SQUARED: [u32; 8] = {
    let r2 = FrConfig::R2.0;
    [
        r2[0] as u32,
        (r2[0] >> 32) as u32,
        r2[1] as u32,
        (r2[1] >> 32) as u32,
        r2[2] as u32,
        (r2[2] >> 32) as u32,
        r2[3] as u32,
        (r2[3] >> 32) as u32,
    ]
};

/// -p⁻¹ mod 2³², used for CIOS Montgomery reduction.
pub const P_INV: u32 = FrConfig::INV as u32;
