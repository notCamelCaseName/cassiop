use std::slice::from_raw_parts;
use ash::{khr::swapchain, khr::portability_subset, Device};
use ash::vk::{ShaderModule, ShaderModuleCreateInfo};
use log::debug;

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
    unsafe {
        let code_u32: &[u32] = std::mem::transmute(code.as_slice());
        debug!("{:?}", code_u32[0]);
        debug!("{:?}", code_u32[1]);
        debug!("{:?}", code_u32[2]);
        debug!("{:?}", code_u32[3]);
        device.create_shader_module(
            &ShaderModuleCreateInfo::default()
                .code(
                    code_u32
                ),
            None
        )
    }.unwrap()
}