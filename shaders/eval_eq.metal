#include "field.h"

/// eval_eq: accumulator[i] += scalar * prod_k factor(bit_k(i), point[k])
///
/// Computes the equality polynomial:
///   eq(X, z) = prod_{k=0}^{n-1} (X_k * z_k + (1 - X_k) * (1 - z_k))
///
/// For each index i, bit k of i determines X_k:
///   bit = 1 => factor = point[k]
///   bit = 0 => factor = (1 - point[k])
///
/// The CPU version (whir) uses the convention where point[0] corresponds
/// to the most significant bit of the index. So:
///   index bit (num_vars-1)  <->  point[0]
///   index bit (num_vars-2)  <->  point[1]
///   ...
///   index bit 0             <->  point[num_vars-1]
///
/// Dispatch with grid_size = 2^num_vars (one thread per accumulator element).
///
/// Buffers:
///   [0] accumulator: device r/w, 2^num_vars field elements (8 u32 limbs each)
///   [1] point:       constant, num_vars field elements
///   [2] scalar:      constant, 1 field element
///   [3] num_vars:    constant, uint (passed via set_bytes — argument buffer, not global memory)

// Max supported num_vars. Threadgroup memory: MAX_VARS * 32 bytes * 2 = 2 KB.
#define MAX_VARS 32

[[kernel]]
void eval_eq_field(
    device uint *accumulator    [[buffer(0)]],
    constant uint *point        [[buffer(1)]],
    constant uint *scalar_buf   [[buffer(2)]],
    constant uint &num_vars     [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]]
) {
    // Precompute point[k] and (1 - point[k]) once per threadgroup.
    // Without this, every thread independently computes field_sub(MONT_ONE, pk)
    // for the same constant values — wasting ALU on identical work.
    threadgroup FieldElement tg_pk[MAX_VARS];
    threadgroup FieldElement tg_one_minus_pk[MAX_VARS];

    // Hoist to register — avoids re-reading from argument buffer each iteration.
    uint nv = num_vars;

    if (lid < nv) {
        tg_pk[lid] = field_load(point, lid);
        tg_one_minus_pk[lid] = field_sub(MONT_ONE, tg_pk[lid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // prod = scalar
    FieldElement prod = field_load(scalar_buf, 0);

    // For each variable k, multiply by point[k] or (1 - point[k]).
    // point[k] maps to bit (nv - 1 - k) of index (matching whir's convention).
    for (uint k = 0; k < nv; k++) {
        uint bit = (index >> (nv - 1 - k)) & 1u;
        FieldElement factor = field_select(tg_pk[k], tg_one_minus_pk[k], bit);
        prod = field_mul(prod, factor);
    }

    // accumulator[index] += prod
    FieldElement acc = field_load_device(accumulator, index);
    acc = field_add(acc, prod);
    field_store(accumulator, index, acc);
}
