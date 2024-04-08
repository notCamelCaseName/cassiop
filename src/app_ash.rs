use {
    crate::{
        debug::{check_validation_layer_support, get_layer_names_and_pointers, setup_debug_messenger, ENABLE_VALIDATION_LAYERS}, surface_info::SurfaceInfo, utility::{self, required_device_extension_names, rusticized_required_device_extension_names}
    },
    anyhow::{Context, Result},
    ash::{
        extensions::{ext::DebugUtils, khr::{Surface, Swapchain}},
        vk,
    },
    ash_window,
    log::*,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    std::{
        cmp,
        collections::HashSet,
        ffi::{c_char, CStr},
        sync::Arc
    },
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

struct Queues {
    graphics_queue: vk::Queue,
    presentation_queue: vk::Queue
}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    presentation_family: Option<u32>
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.presentation_family.is_some()
    }
}

struct SwapchainImage {
    image: vk::Image,
    image_view: vk::ImageView
}

pub struct DoomApp {
    _entry: Arc<ash::Entry>,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    logical_device: ash::Device,
    debug_report_callback: Option<(ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT)>,
    queues: Queues,
    queue_family_indices: QueueFamilyIndices,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    surface_info: SurfaceInfo,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<SwapchainImage>,
}

impl DoomApp {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        debug!("Creating entry");
        let entry = Arc::new(ash::Entry::linked());

        debug!("Creating instance");
        let instance = DoomApp::create_instance(entry.clone(), &window)?;
        debug!("Picking physical device");
        let physical_device = DoomApp::pick_physical_device(&instance)?;
        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
        info!(
            "Using device : {}",
            utility::mnt_to_string(&physical_device_properties.device_name)
        );

        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        debug!("Creating surface");
        let surface_loader = Surface::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )?
        };
        let surface_info = SurfaceInfo::get_surface_info(&surface_loader, &physical_device, &surface)?;

        let queue_family_indices = DoomApp::get_queue_family_indices(&physical_device, &instance, &surface_loader, &surface).expect("No queue family indices found");

        debug!("Creating logical device");
        let (queues, logical_device) = DoomApp::create_logical_device(&instance, &queue_family_indices, &physical_device);

        let swapchain_loader = Swapchain::new(&instance, &logical_device);
        let swapchain = DoomApp::create_swapchain(&swapchain_loader, &surface, &surface_info, &queue_family_indices, &window)?;

        let swapchain_images = DoomApp::get_swapchain_images(&swapchain_loader, &swapchain, &surface_info.choose_best_color_format()?.format, &logical_device)?;

        Ok(Self {
            _entry: entry,
            instance,
            physical_device,
            logical_device,
            debug_report_callback,
            queues,
            queue_family_indices,
            surface,
            surface_loader,
            surface_info,
            swapchain,
            swapchain_loader,
            swapchain_images,
        })
    }

    fn create_instance(entry: Arc<ash::Entry>, window: &window::Window) -> Result<ash::Instance> {
        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"Doom Ash") };
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name)
            .api_version(vk::API_VERSION_1_3);

        let raw_display_handle = window.raw_display_handle();
        let reqs = ash_window::enumerate_required_extensions(raw_display_handle)?;
        let mut req_vec = Vec::from(reqs);

        if ENABLE_VALIDATION_LAYERS {
            req_vec.push(DebugUtils::name().as_ptr());
        }
        
        if cfg!(target_os = "macos") {
            info!("Enabling required extensions for macOS portability.");

            const MACOS_EXT2: [*const c_char; 2] = [
                vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
                vk::KhrPortabilityEnumerationFn::name().as_ptr()
            ];
            req_vec.append(&mut Vec::from(MACOS_EXT2));
        }

        unsafe { Self::validate_required_extensions(reqs, entry.clone())? };

        let flags = if cfg!(target_os = "macos") {
            info!("Enabling instance create flags for macOS portability.");
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let mut create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .flags(flags)
            .enabled_extension_names(req_vec.as_slice());

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(&entry);
            create_info = create_info.enabled_layer_names(&layer_names_ptrs);
        }

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
            debug!("{}", req.to_str().unwrap());
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
                .filter(|&dev| {
                    trace!(
                        "Found physical device : {}",
                        utility::mnt_to_string(
                            &instance.get_physical_device_properties(*dev).device_name
                        )
                    );
                    DoomApp::check_device_extension_support(instance, dev)
                })
                .next()
                .expect("No physical devices supporting required extensions found")
                .to_owned())
        }
    }

    fn get_queue_family_indices(
        device: &vk::PhysicalDevice,
        instance: &ash::Instance,
        surface_loader: &Surface,
        surface: &vk::SurfaceKHR
    ) -> Option<QueueFamilyIndices> {
        let queue_family_properties = unsafe {
            instance.get_physical_device_queue_family_properties(*device)
        };
        
        let mut queue_family_indices = QueueFamilyIndices {graphics_family: None, presentation_family: None};

        queue_family_properties
            .iter()
            .enumerate()
            .for_each(|(i, props)| {
                if props.queue_count > 0 && props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    queue_family_indices.graphics_family = Some(i as u32);
                }

                if let Ok(surface_supported) = unsafe {
                    surface_loader.get_physical_device_surface_support(*device, i as u32, *surface)
                } {
                    if surface_supported {
                        queue_family_indices.presentation_family = Some(i as u32);
                    }
                }
            });

        if queue_family_indices.is_complete() {
            return Some(queue_family_indices)
        }
        None
    }

    fn create_logical_device(
        instance: &ash::Instance,
        queue_family_indices: &QueueFamilyIndices,
        physical_device: &vk::PhysicalDevice,
    ) -> (Queues, ash::Device) {

        let mut indices = vec![
            queue_family_indices.graphics_family.unwrap(),
            queue_family_indices.presentation_family.unwrap()
        ];
        indices.dedup();

        let queue_create_infos = indices
            .iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i as u32)
                .queue_priorities(&[1.])
                .build()
            })
            .collect::<Vec<_>>();

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(required_device_extension_names())
            .build();

        let device = unsafe {
            instance
                .create_device(*physical_device, &create_info, None)
                .unwrap()
        };
        let queues = Queues {
            presentation_queue: unsafe { device.get_device_queue(queue_family_indices.presentation_family.unwrap(), 0) },
            graphics_queue: unsafe { device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0) }
        };

        (queues , device)
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
        rusticized_required_device_extension_names()
            .iter()
            .all(|e| extensions.contains(e))
    }

    fn create_swapchain(
        swapchain_loader: &Swapchain,
        surface: &vk::SurfaceKHR,
        surface_info: &SurfaceInfo,
        queue_family_indices: &QueueFamilyIndices,
        window: &window::Window,
    ) -> Result<vk::SwapchainKHR> {
        let min_image_count = {
            let max_image_count = surface_info.surface_capabilities.max_image_count;

            if max_image_count > 0 {
                cmp::min(
                    max_image_count,
                    surface_info.surface_capabilities.min_image_count + 1,
                )
            } else {
                surface_info.surface_capabilities.min_image_count
            }
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

        let is_concurrent = queue_family_indices.presentation_family != queue_family_indices.graphics_family;
        let queue_family_indices_slice = [
                queue_family_indices.graphics_family.unwrap(),
                queue_family_indices.presentation_family.unwrap()
        ];

        let create_info = if is_concurrent {
            create_info.image_sharing_mode(vk::SharingMode::CONCURRENT)
                       .queue_family_indices(&queue_family_indices_slice)
        } else {
            create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        };

        unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .context("Error while creating swapchain.")
        }
    }

    fn create_image_view(
        image: &vk::Image,
        format: &vk::Format,
        img_aspect_flags: vk::ImageAspectFlags,
        device: &ash::Device,
    ) -> Result<vk::ImageView> {
        let component_mapping_builder = vk::ComponentMapping::builder()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY);
        let img_subresource_range_builder = vk::ImageSubresourceRange::builder()
            .aspect_mask(img_aspect_flags)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let create_info = vk::ImageViewCreateInfo::builder()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(*format)
            .components(*component_mapping_builder)
            .subresource_range(*img_subresource_range_builder);

        unsafe {
            device.create_image_view(&create_info, None)
        }.context("Error occured while trying to create image view")
    }

    fn get_swapchain_images(
        swapchain_loader: &Swapchain,
        swapchain: &vk::SwapchainKHR,
        format: &vk::Format,
        device: &ash::Device,
    ) -> Result<Vec<SwapchainImage>> {
        let swapchain_images = unsafe {
            swapchain_loader.get_swapchain_images(*swapchain)?
        };
        let swapchain_images_output = swapchain_images
            .iter()
            .map(|&image| {
                let image_view = DoomApp::create_image_view(&image, format, vk::ImageAspectFlags::COLOR, device)?;
                Ok::<SwapchainImage, anyhow::Error>(SwapchainImage {image, image_view})
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(swapchain_images_output)
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
            if let Some((utils, messenger)) = self.debug_report_callback.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.swapchain_images.iter().for_each(|img| self.logical_device.destroy_image_view(img.image_view, None));
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.logical_device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
