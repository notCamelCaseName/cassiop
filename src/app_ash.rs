use {
    crate::utility::{self, required_device_extension_names, required_instance_extension_names},
    ash::vk,
    ash_window,
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    winit::{
        event::{Event, ElementState, WindowEvent},
        event_loop::EventLoop,
        keyboard::{Key, NamedKey},
    },
    std::{
        collections::HashSet,
        sync::Arc,
        ffi::CStr,
    },
    log::*,
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
}

impl DoomApp {
    pub fn new(window: &winit::window::Window) -> Self {
        debug!("Creating entry");
        let entry = Arc::new(ash::Entry::linked());

        debug!("Creating instance");
        let instance = DoomApp::create_instance(entry.clone());
        let physical_device = DoomApp::pick_physical_device(&instance);
        debug!("Picking physical device");
        let physical_device_properties = unsafe {instance.get_physical_device_properties(physical_device)};
        info!("Using device : {}", utility::mnt_to_string(&physical_device_properties.device_name));
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
            ).unwrap()
        };


        Self {
            _entry: entry,
            instance,
            physical_device,
            logical_device,
            queue,
            surface,
        }
    }

    fn create_instance(entry: Arc<ash::Entry>) -> ash::Instance {
        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .build();

        let reqs = required_instance_extension_names();

        let flags = if
            cfg!(target_os = "macos")
        {
            info!("Enabling extensions for macOS portability.");
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .flags(flags)
            .enabled_extension_names(&reqs)
            .build();

        unsafe {entry.create_instance(&create_info, None).unwrap()}
    }

    fn pick_physical_device(instance: &ash::Instance) -> vk::PhysicalDevice {
        unsafe {
            instance.enumerate_physical_devices()
                .unwrap()
                .iter()
                .map(|dev| {
                    trace!(
                        "Found physical device : {}",
                        utility::mnt_to_string(&instance.get_physical_device_properties(*dev).device_name)
                    );
                    dev
                })
                .next()
                .unwrap()
                .to_owned()
        }
    }

    fn create_logical_device(instance: &ash::Instance, physical_device: &vk::PhysicalDevice) -> (vk::Queue, ash::Device) {
        let index = unsafe {
            // Get indices of queue families that can do graphics
            instance.get_physical_device_queue_family_properties(*physical_device)
                .iter().enumerate()
                .filter_map(|(i, property)| {
                    if property.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && Self::check_device_extension_support(instance, physical_device) {
                        Some(i)
                    } else {None}
                })
                .next().unwrap()
        };

        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(index as u32)
            .queue_priorities(&[1.])
            .build();

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&[queue_create_info])
            .build();

        let device = unsafe {instance.create_device(*physical_device, &create_info, None).unwrap()};
        let queue = unsafe {device.get_device_queue(index as u32, 0)};

        (queue, device)
    }

    fn check_device_extension_support(instance: &ash::Instance, device: &vk::PhysicalDevice) -> bool {
        let extensions: HashSet<_> = unsafe {
            instance
                .enumerate_device_extension_properties(*device)
                .unwrap()
                .iter()
                .map(|e| String::from_utf8_unchecked(
                        CStr::from_ptr(e.extension_name.as_ptr()).to_bytes().to_vec())
                )
                .collect()
        };
        required_device_extension_names().iter().all(|e| extensions.contains(e))
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

        event_loop.run(move |event, elwt| {

            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => {
                        elwt.exit();
                    },
                    WindowEvent::KeyboardInput { event: input, .. } => {
                        match (input.logical_key, input.state) {
                            (Key::Named(NamedKey::Escape), ElementState::Pressed) => {
                                elwt.exit();
                            },
                            _ => (),
                        }
                    },
                    _ => (),
                }
            }

        }).unwrap()
    }

}

impl Drop for DoomApp {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
