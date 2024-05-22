use ash::{khr::swapchain, khr::portability_subset, Device};
use ash::vk::{ShaderModule, ShaderModuleCreateInfo};
use log::{trace};

/* Unused cfgs, keeping them here for good measure
    #[cfg(target_os = "macos")]
    #[cfg(all(windows))]
    #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
*/

#[cfg(target_os = "macos")]
pub fn rusticized_required_device_extension_names() -> Vec<String> {
    vec![String::from(swapchain::NAME.to_str().unwrap()),
        String::from(portability_subset::NAME.to_str().unwrap())]
}

#[cfg(all(unix, not(target_os = "macos")))]
pub fn rusticized_required_device_extension_names() -> Vec<String> {
    vec![String::from(swapchain::NAME.to_str().unwrap())]
}

#[cfg(target_os = "macos")]
pub fn required_device_extension_names() -> &'static [* const i8] {
    const EXT: [*const i8; 2] = [
        swapchain::NAME.as_ptr(),
        portability_subset::NAME.as_ptr()
    ];
    &EXT
}

#[cfg(all(unix, not(target_os = "macos")))]
pub fn required_device_extension_names() -> &'static [* const i8] {
    const EXT: [*const i8; 1] = [
        swapchain::NAME.as_ptr()
    ];
    &EXT
}

pub fn mnt_to_string(bytes: &[i8]) -> String {
    unsafe { std::str::from_utf8_unchecked(std::mem::transmute(bytes)) }.to_string()
}

pub fn create_shader_module(device: &Device, code: &Vec<u8>) -> ShaderModule {
    let code_u32_vec = &code
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>();
    let code_u32: &[u32] = code_u32_vec.as_slice();
    trace!("code_u32 len in bytes : {}", code_u32.len()*4);
    trace!("code_u32[0] : 0x{:08x?}", code_u32[0]);
    trace!("code_u32[1] : 0x{:08x?}", code_u32[1]);
    trace!("code_u32[2] : 0x{:08x?}", code_u32[2]);
    trace!("code_u32[3] : 0x{:08x?}", code_u32[3]);
    unsafe {
        device.create_shader_module(
            &ShaderModuleCreateInfo::default()
                .code(
                    code_u32
                ),
            None
        )
    }.unwrap()
}
