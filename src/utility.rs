use ash::extensions::khr::Swapchain;

#[cfg(target_os = "macos")]
pub fn required_device_extension_names() -> Vec<String> {
    vec![String::from(Swapchain::name().to_str().unwrap())]
}

#[cfg(all(windows))]
pub fn required_device_extension_names() -> Vec<String> {
    vec![String::from(Swapchain::name().to_str().unwrap())]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_device_extension_names() -> Vec<String> {
    vec![String::from(Swapchain::name().to_str().unwrap())]
}

pub fn mnt_to_string(bytes: &[i8]) -> String {
    unsafe { std::str::from_utf8_unchecked(std::mem::transmute(bytes)) }.to_string()
}
