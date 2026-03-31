mod constants;
mod conversion;

pub use constants::{BN254_MOD, P_INV, P_LIMBS, R_SQUARED};
pub use conversion::Conversion;

pub type Bn254 = ark_bn254::Fr;
