use crate::utility::required_extension_names;
use std::sync::Arc;

use log::{debug, error, log_enabled, info, Level};

use ash::vk;

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

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
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

    pub fn run(self) {
        todo!("No running for now")
    }
}

impl Drop for DoomApp {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
