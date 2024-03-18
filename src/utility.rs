#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
use ash::extensions::khr::XlibSurface;
use ash::extensions::khr::WaylandSurface;
#[cfg(target_os = "macos")]
use ash::extensions::ext::MetalSurface;
#[cfg(target_os = "macos")]
use ash::extensions::khr::GetPhysicalDeviceProperties2;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::Surface;
use ash::extensions::khr::Swapchain;

#[cfg(target_os = "macos")]
pub fn required_extension_names() -> Vec<*const i8> {
    use ash::vk;

    vec![
        Surface::name().as_ptr(),
        MetalSurface::name().as_ptr(),
        GetPhysicalDeviceProperties2::name().as_ptr(),
        vk::KhrPortabilityEnumerationFn::name().as_ptr(),
        DebugUtils::name().as_ptr(),
        Swapchain::name().as_ptr(),
    ]
}

#[cfg(all(windows))]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        Win32Surface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
        Swapchain::name().as_ptr(),
    ]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        WaylandSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
        Swapchain::name().as_ptr(),
    ]
}

pub fn mnt_to_string(bytes: &[i8]) -> String {
    unsafe { std::str::from_utf8_unchecked(std::mem::transmute(bytes)) }.to_string()
}
