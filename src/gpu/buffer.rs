use std::ffi::c_void;

/// Allocate page-aligned memory suitable for zero-copy Metal buffers.
/// Returns a Vec<u32> whose backing allocation is page-aligned.
pub fn alloc_page_aligned(num_u32s: usize) -> Vec<u32> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let byte_size = num_u32s * std::mem::size_of::<u32>();
    let aligned_byte_size = (byte_size + page_size - 1) & !(page_size - 1);
    let aligned_u32s = aligned_byte_size / std::mem::size_of::<u32>();

    unsafe {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let ret = libc::posix_memalign(&mut ptr, page_size, aligned_byte_size);
        assert_eq!(ret, 0, "posix_memalign failed");
        std::ptr::write_bytes(ptr as *mut u8, 0, aligned_byte_size);
        Vec::from_raw_parts(ptr as *mut u32, num_u32s, aligned_u32s)
    }
}

/// Copy data into a new page-aligned Vec<u32>.
pub fn to_page_aligned(data: &[u32]) -> Vec<u32> {
    let mut aligned = alloc_page_aligned(data.len());
    aligned.copy_from_slice(data);
    aligned
}
