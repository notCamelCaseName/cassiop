use {
    crate::{surface_info::SurfaceInfo, utility::{self, required_device_extension_names}},
    anyhow::Result,
    ash::{
        extensions::khr::{
            Surface,
            Swapchain
        },
        vk
    },
    ash_window::{self, enumerate_required_extensions},
    log::*,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    std::{cmp, collections::HashSet, ffi::CStr, sync::Arc},
    winit::{
        event::{ElementState, Event, WindowEvent},
        event_loop::EventLoop,
        keyboard::{Key, NamedKey},
        window,
    },
};

const WINDOW_TITLE: &str = "DoomApp";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

pub struct DoomApp {
    _entry: Arc<ash::Entry>,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    logical_device: ash::Device,
    queue: vk::Queue,
    surface: vk::SurfaceKHR,
    surface_ext: Surface,
    surface_info: SurfaceInfo,
}

impl DoomApp {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        debug!("Creating entry");
        let entry = Arc::new(ash::Entry::linked());

        debug!("Creating instance");
        let instance = DoomApp::create_instance(entry.clone(), &window)?;
        let physical_device = DoomApp::pick_physical_device(&instance)?;
        debug!("Picking physical device");
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
        info!(
            "Using device : {}",
            utility::mnt_to_string(&physical_device_properties.device_name)
        );
        debug!("Creating logical device");
        let (queue, logical_device) = DoomApp::create_logical_device(&instance, &physical_device);
        debug!("Creating surface");
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )?
        };
        let surface_ext = Surface::new(&entry, &instance);
        let surface_info = SurfaceInfo::new(&surface_ext, &physical_device, &surface)?;

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            logical_device,
            queue,
            surface,
            surface_ext,
            surface_info,
        })
    }

    fn create_instance(entry: Arc<ash::Entry>, window: &window::Window) -> Result<ash::Instance> {
        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"Doom Ash") };
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let raw_display_handle = window.raw_display_handle();
        let reqs = enumerate_required_extensions(raw_display_handle)?;
        unsafe { Self::validate_required_extensions(reqs, entry.clone())? };

        let flags = if cfg!(target_os = "macos") {
            info!("Enabling extensions for macOS portability.");
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .flags(flags)
            .enabled_extension_names(&reqs);

        unsafe { Ok(entry.create_instance(&create_info, None)?) }
    }

    unsafe fn validate_required_extensions(
        reqs: &[*const i8],
        entry: Arc<ash::Entry>,
    ) -> Result<()> {
        let instance_ext_properties = entry.enumerate_instance_extension_properties(None)?;
        let available_extensions = instance_ext_properties
            .iter()
            .map(|prop| CStr::from_ptr(prop.extension_name.as_ptr()))
            .collect::<HashSet<_>>();
        let reqs = reqs
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr))
            .collect::<Vec<_>>();
        for req in reqs {
            if !available_extensions.contains(req) {
                return Err(anyhow::anyhow!(format!(
                    "Required extension {} is not available",
                    req.to_str()?
                )));
            }
        }
        Ok(())
    }

    fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice> {
        unsafe {
            Ok(instance
                .enumerate_physical_devices()?
                .iter()
                .map(|dev| {
                    trace!(
                        "Found physical device : {}",
                        utility::mnt_to_string(
                            &instance.get_physical_device_properties(*dev).device_name
                        )
                    );
                    dev
                })
                .next()
                .expect("No physical devices found")
                .to_owned())
        }
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
    ) -> (vk::Queue, ash::Device) {
        let index = unsafe {
            // Get indices of queue families that can do graphics
            instance
                .get_physical_device_queue_family_properties(*physical_device)
                .iter()
                .enumerate()
                .find(|(_i, property)| {
                    property.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && Self::check_device_extension_support(instance, physical_device)
                })
                .expect("First device doesn't support graphics queue")
                .0
        };

        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(index as u32)
            .queue_priorities(&[1.]);

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&[*queue_create_info])
            .build();

        let device = unsafe {
            instance
                .create_device(*physical_device, &create_info, None)
                .unwrap()
        };
        let queue = unsafe { device.get_device_queue(index as u32, 0) };

        (queue, device)
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
    ) -> bool {
        let extensions: HashSet<_> = unsafe {
            instance
                .enumerate_device_extension_properties(*device)
                .unwrap()
                .iter()
                .map(|e| {
                    String::from_utf8_unchecked(
                        CStr::from_ptr(e.extension_name.as_ptr())
                            .to_bytes()
                            .to_vec(),
                    )
                })
                .collect()
        };
        required_device_extension_names()
            .iter()
            .all(|e| extensions.contains(e))
    }


    fn create_swapchain(
        swapchain_ext: &Swapchain,
        surface: &vk::SurfaceKHR,
        surface_info: &SurfaceInfo,
        queue: &vk::Queue, // a QueueFamilyIndices struct seems to be needed
        window: &window::Window,
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
    ) -> Result<vk::SwapchainKHR> {
        let min_image_count = cmp::min(
            surface_info.surface_capabilities.max_image_count,
            surface_info.surface_capabilities.min_image_count + 1
        );
        let queue_family_properties = unsafe {
            instance.get_physical_device_queue_family_properties(*device)
        };

        let current_transform = surface_info.surface_capabilities.current_transform;
        let best_format = surface_info.choose_best_color_format()?;
        let swapchain_extent = surface_info.choose_swapchain_extents(window)?;

        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(*surface)
            .min_image_count(min_image_count)
            .image_format(best_format.format)
            .image_color_space(best_format.color_space)
            .image_extent(swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(surface_info.choose_best_pres_mode()?)
            .clipped(true);

        /*
        let queue_family_indices = [
            queue_family_indices.graphics_family.unwrap(),
            queue_family_indices.presentation_family.unwrap(),
        ];
        let is_concurrent = queue_family_indices[0] != queue_family_indices[1];

        let create_info = if is_concurrent {
            create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices)
        } else {
            create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        */
        todo!()

    }

    pub fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        let window = winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Couldn't create window.");

        window
    }

    pub fn main_loop(self, event_loop: EventLoop<()>) {
        event_loop
            .run(move |event, elwt| {
                if let Event::WindowEvent { event, .. } = event {
                    match event {
                        WindowEvent::CloseRequested => {
                            elwt.exit();
                        }
                        WindowEvent::KeyboardInput { event: input, .. } => {
                            match (input.logical_key, input.state) {
                                (Key::Named(NamedKey::Escape), ElementState::Pressed) => {
                                    elwt.exit();
                                }
                                _ => (),
                            }
                        }
                        _ => (),
                    }
                }
            })
            .unwrap()
    }
}

impl Drop for DoomApp {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.destroy_device(None);
            self.surface_ext.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
