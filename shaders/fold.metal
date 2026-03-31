#include "field.h"

/// Fold: linear interpolation between low and high halves of a buffer.
///
/// Given a buffer of 2*half_size field elements and a weight:
///   values[i] += (values[half_size + i] - values[i]) * weight
///
/// After this kernel, only the first half_size elements are meaningful.
/// The caller is responsible for logically truncating.
///
/// Dispatch with grid_size = half_size (one thread per output element).
/// Guaranteed: index < half_size, so values[half_size + index] is in-bounds.
///
/// Buffers:
///   [0] values:     device r/w, at least 2 * half_size field elements
///   [1] weight:     constant, 1 field element
///   [2] half_size:  constant, uint (passed via set_bytes — argument buffer)
[[kernel]]
void fold_field(
    device uint *values        [[buffer(0)]],
    constant uint *weight_buf  [[buffer(1)]],
    constant uint &half_size   [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    FieldElement w = field_load(weight_buf, 0);

    // Both reads are necessary: lo appears in the subtraction AND the addition.
    // result = lo + (hi - lo) * w  (linear interpolation)
    FieldElement lo = field_load_device(values, index);
    FieldElement hi = field_load_device(values, half_size + index);

    FieldElement diff = field_sub(hi, lo);
    FieldElement scaled = field_mul(diff, w);
    FieldElement result = field_add(lo, scaled);

    field_store(values, index, result);
}
