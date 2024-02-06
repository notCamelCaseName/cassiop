use crate::utility::required_extension_names;
use crate::debug::{check_validation_layer_support, get_layer_names_and_pointers, setup_debug_messenger, ENABLE_VALIDATION_LAYERS};

use std::ffi::CStr;
use std::sync::Arc;

use ash::extensions::ext::DebugUtils;
use log::{debug, info};

use ash::vk;
use winit::event::{Event, VirtualKeyCode, ElementState, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};

const WINDOW_TITLE: &str = "DoomApp";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

pub struct DoomApp {
    _entry: Arc<ash::Entry>,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    debug_report_callback: Option<(ash::extensions::ext::DebugUtils, vk::DebugUtilsMessengerEXT)>
}

impl DoomApp {
    pub fn new() -> Self {
        debug!("Creating entry");
        let entry = Arc::new(ash::Entry::linked());
        debug!("Creating instance");
        let instance = DoomApp::create_instance(entry.clone());

        let physical_device = DoomApp::pick_physical_device(&instance);
        let debug_report_callback = setup_debug_messenger(&entry, &instance);

        Self {
            _entry: entry,
            instance,
            physical_device,
            debug_report_callback
        }
    }

    fn create_instance(entry: Arc<ash::Entry>) -> ash::Instance {
        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .build();

        let mut reqs = required_extension_names();
        if ENABLE_VALIDATION_LAYERS {
            reqs.push(DebugUtils::name().as_ptr());
        }

        let flags = if 
            cfg!(target_os = "macos")
        {
            info!("Enabling extensions for macOS portability.");
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let mut create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .flags(flags)
            .enabled_extension_names(&reqs);

        let (_layer_names, layer_names_ptrs) = get_layer_names_and_pointers();

        if ENABLE_VALIDATION_LAYERS {
            check_validation_layer_support(&entry);
            create_info = create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe {entry.create_instance(&create_info.build(), None).unwrap()}
    }

    fn pick_physical_device(instance: &ash::Instance) -> vk::PhysicalDevice {
        let devices = unsafe { instance.enumerate_physical_devices().unwrap() };
        let device = devices
            .into_iter()
            .find(|device| Self::is_device_suitable(instance, *device))
            .expect("No suitable physical device.");

        let props = unsafe { instance.get_physical_device_properties(device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });
        device
    }

    fn is_device_suitable(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
        Self::find_queue_families(instance, device).is_some()
    }

    fn find_queue_families(instance: &ash::Instance, device: vk::PhysicalDevice) -> Option<usize> {
        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        props
            .iter()
            .enumerate()
            .find(|(_, family)| {
                family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .map(|(index, _)| index)
    }

    pub fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Couldn't create window.")
    }

    pub fn main_loop(event_loop: EventLoop<()>) {

        event_loop.run(move |event, _, control_flow| {

            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit
                    },
                    WindowEvent::KeyboardInput { input, .. } => {
                        match (input.virtual_keycode, input.state) {
                            (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                dbg!();
                                *control_flow = ControlFlow::Exit
                            },
                            _ => (),
                        }
                    },
                    _ => (),
                }
            }

        })
    }

    pub fn run(self) {
        let event_loop = EventLoop::new();
        let _window = DoomApp::init_window(&event_loop);

        DoomApp::main_loop(event_loop);
    }
}

impl Drop for DoomApp {
    fn drop(&mut self) {
        unsafe {
            if let Some((utils, messenger)) = self.debug_report_callback.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
