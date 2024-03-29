use {
    anyhow::{
        anyhow, Result
    },
    ash::{
        extensions::khr::{
            Surface,
            Swapchain
        },
        vk,
    },
    log::info,
    std::cmp,
    winit::window
};

pub struct SurfaceInfo {
    pub present_modes: Vec<vk::PresentModeKHR>,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
}

impl SurfaceInfo {
    pub fn new(
        surface_ext: &Surface,
        physical_device: &vk::PhysicalDevice,
        surface: &vk::SurfaceKHR
    ) -> Result<Self> {
        let present_modes = unsafe {
            (*surface_ext).get_physical_device_surface_present_modes(*physical_device, *surface)
        }?;
        let surface_capabilities = unsafe {
            (*surface_ext).get_physical_device_surface_capabilities(*physical_device, *surface)
        }?;
        let surface_formats = unsafe {
            (*surface_ext).get_physical_device_surface_formats(*physical_device, *surface)
        }?;
        Ok(SurfaceInfo {
            present_modes,
            surface_capabilities,
            surface_formats,
        })
    }

    pub fn choose_best_color_format(&self) -> Result<vk::SurfaceFormatKHR> {
        const DESIRED_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
        const DESIRED_FORMAT_ALT: vk::Format = vk::Format::B8G8R8A8_UNORM;
        const DESIRED_COLOR_SPACE: vk::ColorSpaceKHR = vk::ColorSpaceKHR::SRGB_NONLINEAR;

        if self.surface_formats.len() == 1 && self.surface_formats[0].format == vk::Format::UNDEFINED {
            // UNDEFINED means all formats are supported
            Ok(vk::SurfaceFormatKHR {
                format: DESIRED_FORMAT,
                color_space: DESIRED_COLOR_SPACE,
            })
        } else {
            if let Some(res) = self.surface_formats.iter().find(|sf| {
                (sf.format == DESIRED_FORMAT || sf.format == DESIRED_FORMAT_ALT)
                    && sf.color_space == DESIRED_COLOR_SPACE
            }) {
                Ok(*res)
            } else {
                Err(anyhow!("None of the desired color formats or color spaces are supported by the physical device."))
            }
        }
    }

    pub fn choose_best_pres_mode(&self) -> Result<vk::PresentModeKHR> {
        const DESIRED_MODE: vk::PresentModeKHR = vk::PresentModeKHR::MAILBOX;
        if self.present_modes.iter().any(|&mode| mode==DESIRED_MODE) {
            Ok(DESIRED_MODE)
        } else {
            Ok(vk::PresentModeKHR::FIFO)
                // FIFO mode cannot be absent according to the Vulkan spec
        }
    }

    pub fn choose_swapchain_extents(&self, window: &window::Window) -> Result<vk::Extent2D> {
        let current_extent = self.surface_capabilities.current_extent;
        
        if current_extent.width != u32::MAX {
            Ok(current_extent)
        } else {
            info!("Can't get current_extent from surface ! Getting it through winit...");
            let window_size = window.inner_size();
            let (width, height) = (window_size.width, window_size.height);
            let (min_width, min_height) = (
                self.surface_capabilities.min_image_extent.width,
                self.surface_capabilities.min_image_extent.height,
            );
            let (max_width, max_height) = (
                self.surface_capabilities.max_image_extent.width,
                self.surface_capabilities.max_image_extent.height,
            );
            let width = cmp::min(cmp::max(width, min_width), max_width);
            let height = cmp::min(cmp::max(height, min_height), max_height);
            Ok(vk::Extent2D { width, height })
        }
    }
}
