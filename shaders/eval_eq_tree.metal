#include "field.h"

/// Tree-based eval_eq expansion: one level of the binary tree.
///
/// At level k, the source buffer has 2^k valid elements.
/// Each thread reads src[tid], computes:
///   hi = src[tid] * point[level]
///   lo = src[tid] - hi                 (= src[tid] * (1 - point[level]))
/// and writes:
///   dst[2*tid]     = lo
///   dst[2*tid + 1] = hi
///
/// After this pass, dst has 2^(k+1) valid elements.
///
/// Uses hi = val * pk, lo = val - hi instead of two multiplies.
/// This halves the multiply count: 1 field_mul + 1 field_sub per thread
/// instead of 2 field_muls.
///
/// Dispatch with grid_size = 2^level (one thread per source element).
///
/// Buffers:
///   [0] src:    device read, 2^level field elements
///   [1] dst:    device write, 2^(level+1) field elements
///   [2] point:  constant, array of field elements
///   [3] level:  constant, uint — index into point array
[[kernel]]
void eval_eq_tree_expand(
    device uint *src           [[buffer(0)]],
    device uint *dst           [[buffer(1)]],
    constant uint *point       [[buffer(2)]],
    constant uint &level       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    FieldElement pk = field_load(point, level);
    FieldElement val = field_load_device(src, tid);

    FieldElement hi = field_mul(val, pk);
    FieldElement lo = field_sub(val, hi);

    field_store(dst, 2 * tid,     lo);
    field_store(dst, 2 * tid + 1, hi);
}
