use crate::utility::required_extension_names;

use ash::vk;

pub struct DoomApp {
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
}

impl DoomApp {
    pub fn new() -> Self {
        let entry = ash::Entry::linked();
        let instance = DoomApp::create_instance(entry);

        let physical_device = DoomApp::pick_physical_device(&instance);

        Self {
            instance,
            physical_device,
        }
    }

    fn create_instance(entry: ash::Entry) -> ash::Instance {
        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .build();
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&required_extension_names())
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
