#ifndef FIELD_H
#define FIELD_H

#include <metal_stdlib>
using namespace metal;

// ============================================================
// BN254 scalar field arithmetic in Montgomery form
// ============================================================

struct FieldElement {
    uint limbs[8]; // 8 x 32-bit limbs (little-endian)
};

// BN254 scalar field modulus p (little-endian u32 limbs)
constant FieldElement P = {{
    0xF0000001u,
    0x43E1F593u,
    0x79B97091u,
    0x2833E848u,
    0x8181585Du,
    0xB85045B6u,
    0xE131A029u,
    0x30644E72u
}};

// -p^{-1} mod 2^{32}, for CIOS Montgomery reduction
constant uint P_INV = 0xEFFFFFFFu;

// Montgomery representation of 1: R mod p where R = 2^{256}
constant FieldElement MONT_ONE = {{
    0x4FFFFFFBu,
    0xAC96341Cu,
    0x9F60CD29u,
    0x36FC7695u,
    0x7879462Eu,
    0x666EA36Fu,
    0x9A07DF2Fu,
    0x0E0A77C1u
}};

// Returns true if a >= P
inline bool check_gte(FieldElement a) {
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] > P.limbs[i]) return true;
        if (a.limbs[i] < P.limbs[i]) return false;
    }
    return true; // equal
}

// Computes a - P, assuming a >= P (pure 32-bit)
inline FieldElement subtract_P(FieldElement a) {
    FieldElement result;
    uint borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint ai = a.limbs[i];
        uint pi = P.limbs[i];
        uint diff = ai - pi;
        uint b1 = (ai < pi) ? 1u : 0u;
        result.limbs[i] = diff - borrow;
        uint b2 = (diff < borrow) ? 1u : 0u;
        borrow = b1 + b2;
    }
    return result;
}

// Modular addition: (a + b) mod P — pure 32-bit carry chain
inline FieldElement field_add(FieldElement a, FieldElement b) {
    FieldElement result;
    uint carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint s = a.limbs[i] + b.limbs[i];
        uint c = (s < a.limbs[i]) ? 1u : 0u;
        uint r = s + carry;
        c += (r < s) ? 1u : 0u;
        result.limbs[i] = r;
        carry = c;
    }
    if (check_gte(result)) {
        result = subtract_P(result);
    }
    return result;
}

// Modular subtraction: (a - b) mod P — pure 32-bit borrow chain
inline FieldElement field_sub(FieldElement a, FieldElement b) {
    FieldElement result;
    uint borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint ai = a.limbs[i];
        uint bi = b.limbs[i];
        uint diff = ai - bi;
        uint b1 = (ai < bi) ? 1u : 0u;
        result.limbs[i] = diff - borrow;
        uint b2 = (diff < borrow) ? 1u : 0u;
        borrow = b1 + b2;
    }
    // If underflow, add P back (pure 32-bit)
    if (borrow) {
        uint carry = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint s = result.limbs[i] + P.limbs[i];
            uint c = (s < result.limbs[i]) ? 1u : 0u;
            uint r = s + carry;
            c += (r < s) ? 1u : 0u;
            result.limbs[i] = r;
            carry = c;
        }
    }
    return result;
}

// Montgomery multiplication using CIOS algorithm
// Computes: a * b * R^{-1} mod P  (where R = 2^{256})
inline FieldElement field_mul(FieldElement a, FieldElement b) {
    ulong T[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        // Part A: T += a[i] * b
        ulong carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            ulong product = (ulong)a.limbs[i] * (ulong)b.limbs[j];
            ulong sum = T[j] + product + carry;
            T[j] = sum & 0xFFFFFFFF;
            carry = sum >> 32;
        }
        T[8] += carry;

        // Part B: Montgomery reduction
        uint m = (uint)T[0] * P_INV;

        carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            ulong product = (ulong)m * (ulong)P.limbs[j];
            ulong sum = T[j] + product + carry;
            T[j] = sum & 0xFFFFFFFF;
            carry = sum >> 32;
        }
        T[8] += carry;

        // Shift right
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            T[j] = T[j + 1];
        }
        T[8] = 0;
    }

    FieldElement result;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = (uint)T[i];
    }

    if (check_gte(result)) {
        result = subtract_P(result);
    }

    return result;
}

// Branchless select: returns a if condition != 0, b otherwise
inline FieldElement field_select(FieldElement a, FieldElement b, uint condition) {
    FieldElement result;
    uint mask = 0u - condition; // 0x00000000 or 0xFFFFFFFF
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        result.limbs[i] = (a.limbs[i] & mask) | (b.limbs[i] & ~mask);
    }
    return result;
}

// Load a field element from a uint buffer at a given element index
inline FieldElement field_load(const constant uint *buf, uint element_index) {
    FieldElement e;
    uint offset = element_index * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        e.limbs[i] = buf[offset + i];
    }
    return e;
}

// Load a field element from a device (read+write) buffer
inline FieldElement field_load_device(const device uint *buf, uint element_index) {
    FieldElement e;
    uint offset = element_index * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        e.limbs[i] = buf[offset + i];
    }
    return e;
}

// Store a field element into a device buffer at a given element index
inline void field_store(device uint *buf, uint element_index, FieldElement e) {
    uint offset = element_index * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        buf[offset + i] = e.limbs[i];
    }
}

#endif // FIELD_H
