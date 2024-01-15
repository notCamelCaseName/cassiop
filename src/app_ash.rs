use crate::utility::required_extension_names;
use std::sync::Arc;

use log::{debug, info};

use ash::vk;
use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};

const WINDOW_TITLE: &'static str = "DoomApp";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

pub struct DoomApp {
    _entry: Arc<ash::Entry>,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
}

impl DoomApp {
    pub fn new() -> Self {
        debug!("Creating entry");
        let entry = Arc::new(ash::Entry::linked());
        debug!("Creating instance");
        let instance = DoomApp::create_instance(entry.clone());

        let physical_device = DoomApp::pick_physical_device(&instance);

        Self {
            _entry: entry,
            instance,
            physical_device,
        }
    }

    fn create_instance(entry: Arc<ash::Entry>) -> ash::Instance {
        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .build();

        let reqs = required_extension_names();

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
                .get(0)
                .unwrap()
                .to_owned()
        }
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

            match event {
                | Event::WindowEvent { event, .. } => {
                    match event {
                        | WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit
                        },
                        | WindowEvent::KeyboardInput { input, .. } => {
                            match input {
                                | KeyboardInput { virtual_keycode, state, .. } => {
                                    match (virtual_keycode, state) {
                                        | (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                            dbg!();
                                            *control_flow = ControlFlow::Exit
                                        },
                                        | _ => {},
                                    }
                                },
                            }
                        },
                        | _ => {},
                    }
                },
                _ => (),
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
            self.instance.destroy_instance(None);
        }
    }
}
