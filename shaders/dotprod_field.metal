#include "field.h"

[[kernel]]
void dot_product_field(
    constant uint *inA [[buffer(0)]],
    constant uint *inB [[buffer(1)]],
    device uint *result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    FieldElement a = field_load(inA, index);
    FieldElement b = field_load(inB, index);
    FieldElement res = field_mul(a, b);
    field_store(result, index, res);
}

// accumulator[i] += weight * vector[i]
[[kernel]]
void scalar_mul_add_field(
    device uint *accumulator [[buffer(0)]],      // read+write
    constant uint *weight    [[buffer(1)]],       // single element (8 limbs)
    constant uint *vector    [[buffer(2)]],       // read-only
    uint index [[thread_position_in_grid]]
) {
    FieldElement w = field_load(weight, 0);
    FieldElement v = field_load(vector, index);
    FieldElement acc = field_load_device(accumulator, index);

    acc = field_add(acc, field_mul(w, v));

    field_store(accumulator, index, acc);
}
