use ash::{extensions::khr::Swapchain, vk};

/* Unused cfgs, keeping them here for good measure
    #[cfg(target_os = "macos")]
    #[cfg(all(windows))]
    #[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
*/

#[cfg(target_os = "macos")]
pub fn rusticized_required_device_extension_names() -> Vec<String> {
    vec![String::from(Swapchain::name().to_str().unwrap()),
        String::from(vk::KhrPortabilitySubsetFn::name().to_str().unwrap())]
}

#[cfg(all(unix, not(target_os = "macos")))]
pub fn rusticized_required_device_extension_names() -> Vec<String> {
    vec![String::from(vk::KhrPortabilitySubsetFn::name().to_str().unwrap())]
}

#[cfg(target_os = "macos")]
pub fn required_device_extension_names() -> &'static [* const i8] {
    const EXT: [*const i8; 2] = [
        Swapchain::name().as_ptr(),
        vk::KhrPortabilitySubsetFn::name().as_ptr()
    ];
    &EXT
}

#[cfg(all(unix, not(target_os = "macos")))]
pub fn required_device_extension_names() -> &'static [* const i8] {
    const EXT: [*const i8; 2] = [
        Swapchain::name().as_ptr()
    ];
    &EXT
}

pub fn mnt_to_string(bytes: &[i8]) -> String {
    unsafe { std::str::from_utf8_unchecked(std::mem::transmute(bytes)) }.to_string()
}
