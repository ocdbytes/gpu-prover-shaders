# Metal learn

Metal learning repo

## Shaders

file : `shaders/dotprod.metal`
```metal
// "kernel" keywork means that this is a compute function
[[kernel]]
void dot_product(
  // "constant" means that this data is available for GPU to only read.
  constant uint *inA [[buffer(0)]],
  // "buffer(id)" : "id" specifies the indexing of the buffer
  // buffer is a shared memory space between CPU and GPU.
  // 1, 2, 3 we are providing will be used as a reference by the kernel to access or write the data.
  constant uint *inB [[buffer(1)]],
  // "device" means that this variable is accessible for GPU to read and write to.
  device uint *result [[buffer(2)]],
  // we provide the index through gridding the operation.
  uint index [[thread_position_in_grid]])
{
  result[index] = inA[index] * inB[index];
}
```

### Compilation

```sh
xcrun -sdk macosx metal -c dotprod.metal -o dotprod.air
xcrun -sdk macosx metallib dotprod.air -o dotprod.metallib
```

## Run Benches

```sh
cargo run --release
```
