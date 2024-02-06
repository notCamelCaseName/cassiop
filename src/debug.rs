use ash::extensions::ext::DebugUtils;
use ash::vk::DebugUtilsMessengerCreateFlagsEXT;
use ash::{vk, Entry, Instance};
use std::ffi::{CStr, c_char, CString};
use std::os::raw::c_void;

// https://github.com/adrien-ben/vulkan-tutorial-rs

pub const ENABLE_VALIDATION_LAYERS: bool = true;
pub const REQUIRED_LAYERS: [&'static str; 1] = ["VK_LAYER_LUNARG_standard_validation"];

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;

    let message = CStr::from_ptr((*p_callback_data).p_message);
    match flag {
        Flag::VERBOSE => log::debug!("{:?} - {:?}", typ, message),
        Flag::INFO => log::info!("{:?} - {:?}", typ, message),
        Flag::WARNING => log::warn!("{:?} - {:?}", typ, message),
        _ => log::error!("{:?} - {:?}", typ, message),
    }
    vk::FALSE
}

pub fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Option<(DebugUtils, vk::DebugUtilsMessengerEXT)> {
    if !ENABLE_VALIDATION_LAYERS {
        return None;
    }

    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .flags(DebugUtilsMessengerCreateFlagsEXT::from_raw(u32::MAX))
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::from_raw(u32::MAX))
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::from_raw(u32::MAX))
        .pfn_user_callback(Some(vulkan_debug_callback));
    let debug_utils: DebugUtils = DebugUtils::new(entry, instance);
    let debug_utils_messenger = unsafe {
        debug_utils
            .create_debug_utils_messenger(&create_info, None)
            .unwrap()
    };

    Some((debug_utils, debug_utils_messenger))
}

pub fn check_validation_layer_support(entry: &Entry) {
    for required in REQUIRED_LAYERS.iter() {
        let found = entry
            .enumerate_instance_layer_properties()
            .unwrap()
            .iter()
            .any(|layer| {
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                let name = name.to_str().expect("Failed to get layer name pointer");
                required == &name
            });

        if !found {
            panic!("Validation layer not supported: {}", required);
        }
    }
}

pub fn get_layer_names_and_pointers() -> (Vec<CString>, Vec<*const c_char>) {
    let layer_names = REQUIRED_LAYERS
        .iter()
        .map(|name| CString::new(*name).unwrap())
        .collect::<Vec<_>>();
    let layer_names_ptrs = layer_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();
    (layer_names, layer_names_ptrs)
}