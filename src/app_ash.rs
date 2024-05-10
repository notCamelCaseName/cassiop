use std::collections::HashMap;
use std::mem;
use ash::{Device, khr};
use ash::prelude::VkResult;
use {
    crate::{
        debug::{check_validation_layer_support, get_layer_names_and_pointers, setup_debug_messenger, ENABLE_VALIDATION_LAYERS}, surface_info::SurfaceInfo, utility::{self, required_device_extension_names, rusticized_required_device_extension_names}
    },
    anyhow::{Context, Result},
    ash::{
        ext::debug_utils,
        khr::{get_physical_device_properties2, portability_enumeration, surface, swapchain},
        vk::{self, ShaderModule},
    },
    ash_window,
    log::*,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
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
use crate::utility::create_shader_module;

const WINDOW_TITLE: &str = "DoomApp";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

const MAX_FRAMES: usize = 2;

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

#[repr(C)]
struct Vertex {
    position: [f32; 3], // offset 0
    color: [f32; 3],    // offset 12
}

pub struct DoomApp
{
    entry: Arc<ash::Entry>,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    debug_report_callback: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    queues: Queues,
    queue_family_indices: QueueFamilyIndices,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    surface_info: SurfaceInfo,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<SwapchainImage>,
    swapchain_extent: vk::Extent2D,
    shader_modules: HashMap<String, ShaderModule>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    queue_submit_complete_semaphores: Vec<vk::Semaphore>,
    queue_submit_complete_fences: Vec<vk::Fence>,
    window: Box<window::Window>
}

impl DoomApp
{
    pub fn new(window: Box<window::Window>) -> Result<Self>
    {
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
        let surface_loader = surface::Instance::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().into(),
                window.window_handle().unwrap().into(),
                None,
            )?
        };
        let surface_info = SurfaceInfo::get_surface_info(&surface_loader, &physical_device, &surface)?;

        let queue_family_indices = DoomApp::get_queue_family_indices(&physical_device, &instance, &surface_loader, &surface).expect("No queue family indices found");

        debug!("Creating logical device");
        let (queues, device) = DoomApp::create_logical_device(&instance, &queue_family_indices, &physical_device);

        debug!("Creating swapchain");
        let swapchain_loader = swapchain::Device::new(&instance, &device);
        let swapchain_extent = surface_info.surface_capabilities.current_extent;
        let swapchain = DoomApp::create_swapchain(
            &swapchain_loader,
            &surface,
            &surface_info,
            &queue_family_indices,
            &swapchain_extent
        )?;

        let swapchain_images = DoomApp::get_swapchain_images(&swapchain_loader, &swapchain, &surface_info.choose_best_color_format()?.format, &device)?;

        debug!("Loading shaders");
        let mut shader_modules: HashMap<String, ShaderModule> = HashMap::new();
        for elt in std::fs::read_dir("shaders/")? {
            let elt = elt?;
            if elt.path().extension().unwrap().to_str() == Some("spv") {
                let shader_bin = std::fs::read(elt.path())?;
                shader_modules.insert(elt.file_name().into_string().unwrap(), create_shader_module(&device, &shader_bin));
            }
        }

        let pipeline_layout = Self::create_pipeline_layout(&device)?;

        let render_pass = Self::create_render_passe(&device, &surface_info)?;

        let pipeline = Self::create_graphics_pipeline(
            *shader_modules.get("triangle.vert.spv").unwrap(),
            *shader_modules.get("triangle.frag.spv").unwrap(),
            swapchain_extent,
            pipeline_layout,
            render_pass,
            &device
        )?;

        let framebuffers = swapchain_images
            .iter()
            .map(|image| {
                Self::create_framebuffer(
                    &device,
                    render_pass,
                    image.image_view,
                    swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let command_pool = Self::create_command_pool(
            &device,
            queue_family_indices.graphics_family.unwrap()
        )?;

        let command_buffers = Self::allocate_command_buffers(
            &device,
            command_pool,
            framebuffers.len() as u32
        )?;

        unsafe {
            Self::record_command_buffers(
                &device,
                command_buffers.as_slice(),
                framebuffers.as_slice(),
                render_pass,
                swapchain_extent,
                pipeline
            )?
        };

        let (
            image_available_semaphores,
            queue_submit_complete_semaphores,
            queue_submit_complete_fences,
        ) = Self::create_synchronization(&device, MAX_FRAMES)?;

        let vertices = [
            Vertex {
                position: [0.0,0.0,0.0],
                color: [0.0,0.0,0.0]
            }
        ];

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            debug_report_callback,
            queues,
            queue_family_indices,
            surface,
            surface_loader,
            surface_info,
            swapchain,
            swapchain_loader,
            swapchain_images,
            swapchain_extent,
            shader_modules,
            render_pass,
            pipeline_layout,
            pipeline,
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            queue_submit_complete_semaphores,
            queue_submit_complete_fences,
            window
        })
    }

    fn create_instance(entry: Arc<ash::Entry>, window: &window::Window) -> Result<ash::Instance>
    {
        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"Doom Ash") };
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .api_version(vk::API_VERSION_1_3);

        let display_handle = window.display_handle().unwrap();
        let reqs = ash_window::enumerate_required_extensions(display_handle.into())?;
        let mut req_vec = Vec::from(reqs);

        if ENABLE_VALIDATION_LAYERS {
            req_vec.push(debug_utils::NAME.as_ptr());
        }
        
        if cfg!(target_os = "macos") {
            info!("Enabling required extensions for macOS portability.");

            const MACOS_EXT2: [*const c_char; 2] = [
                get_physical_device_properties2::NAME.as_ptr(),
                portability_enumeration::NAME.as_ptr()
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

        let mut create_info = vk::InstanceCreateInfo::default()
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
    ) -> Result<()>
    {
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

    fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice>
    {
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
        surface_loader: &surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> Option<QueueFamilyIndices>
    {
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
    ) -> (Queues, Device)
    {

        let mut indices = vec![
            queue_family_indices.graphics_family.unwrap(),
            queue_family_indices.presentation_family.unwrap()
        ];
        indices.dedup();

        let queue_create_infos = indices
            .iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::default()
                .queue_family_index(*i)
                .queue_priorities(&[1.])
            })
            .collect::<Vec<_>>();

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(required_device_extension_names());

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
    ) -> bool
    {
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
        swapchain_loader: &swapchain::Device,
        surface: &vk::SurfaceKHR,
        surface_info: &SurfaceInfo,
        queue_family_indices: &QueueFamilyIndices,
        swapchain_extent: &vk::Extent2D,
    ) -> Result<vk::SwapchainKHR>
    {
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

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(min_image_count)
            .image_format(best_format.format)
            .image_color_space(best_format.color_space)
            .image_extent(*swapchain_extent)
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
        device: &Device,
    ) -> Result<vk::ImageView>
    {
        let component_mapping_builder = vk::ComponentMapping::default()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY);
        let img_subresource_range_builder = vk::ImageSubresourceRange::default()
            .aspect_mask(img_aspect_flags)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let create_info = vk::ImageViewCreateInfo::default()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(*format)
            .components(component_mapping_builder)
            .subresource_range(img_subresource_range_builder);

        unsafe {
            device.create_image_view(&create_info, None)
        }.context("Error occurred while trying to create image view")
    }

    fn get_swapchain_images(
        swapchain_loader: &swapchain::Device,
        swapchain: &vk::SwapchainKHR,
        format: &vk::Format,
        device: &Device,
    ) -> Result<Vec<SwapchainImage>>
    {
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

    fn create_pipeline_layout(device: &Device) -> Result<vk::PipelineLayout>
    {
        unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[])
                    .push_constant_ranges(&[]),
                None,
            )
        }.context("Error trying to create a pipeline layout.")
    }

    fn create_render_passe(
        device: &Device,
        surface_info: &SurfaceInfo,
    ) -> VkResult<vk::RenderPass>
    {
        let attachment_descriptions = [vk::AttachmentDescription::default()
            .format(surface_info.choose_best_color_format().unwrap().format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

        let attachment_references = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
        let subpass_descriptions = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&attachment_references)];

        let subpass_dependencies = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        unsafe {device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachment_descriptions)
                .subpasses(&subpass_descriptions)
                .dependencies(&subpass_dependencies),
            None,
        ) }
    }

    fn create_graphics_pipeline(
        vertex_shader: ShaderModule,
        fragment_shader: ShaderModule,
        swapchain_extents: vk::Extent2D,
        pipeline_layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        device: &Device,
    ) -> Result<vk::Pipeline>
    {
        let name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_shader)
                .name(name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_shader)
                .name(name),
        ];

        let binding_descriptions = [vk::VertexInputBindingDescription::default()
            .stride(mem::size_of::<Vertex>().try_into().unwrap())
            .input_rate(vk::VertexInputRate::VERTEX)
        ];

        let attribute_descriptions = [vk::VertexInputAttributeDescription::default().format(vk::Format::R32G32B32_SFLOAT),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12)
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewports = [vk::Viewport {
            width: swapchain_extents.width as f32,
            height: swapchain_extents.height as f32,
            max_depth: 1.0,
            ..vk::Viewport::default()
        }];
        let scissors = [vk::Rect2D {
            extent: swapchain_extents,
            ..vk::Rect2D::default()
        }];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
        let graphics_pipeline_create_infos = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)];
        unsafe {
            let pipelines = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &graphics_pipeline_create_infos, None)
                .map_err(|(_, e)| e)?;
            Ok(pipelines[0])
        }
    }

    fn create_framebuffer(
        device: &Device,
        render_pass: vk::RenderPass,
        image_view: vk::ImageView,
        swapchain_extent: vk::Extent2D,
    ) -> Result<vk::Framebuffer>
    {
        let attachments = [image_view];
        unsafe {
            device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(render_pass)
                        .attachments(&attachments)
                        .width(swapchain_extent.width)
                        .height(swapchain_extent.height)
                        .layers(1),
                    None,
                )
                .context("Failed to create a framebuffer.")
        }
    }

    fn create_command_pool(device: &Device, queue_family_index: u32) -> Result<vk::CommandPool>
    {
        unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index),
                    None,
                )
                .context("Failed to create a command pool.")
        }
    }

    fn allocate_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
        buffer_count: u32,
    ) -> Result<Vec<vk::CommandBuffer>>
    {
        unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(buffer_count),
                )
                .context("Failed to allocate command buffers.")
        }
    }

    unsafe fn record_command_buffers(
        device: &Device,
        command_buffers: &[vk::CommandBuffer],
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        swapchain_extent: vk::Extent2D,
        graphics_pipeline: vk::Pipeline,
    ) -> Result<()>
    {
        for (command_buffer, framebuffer) in command_buffers.into_iter().zip(framebuffers) {
            device
                .begin_command_buffer(
                    *command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
                )
                .context("Failed to begin command buffer.")?;
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0., 0., 0. , 1.0],
                },
            }];
            device.cmd_begin_render_pass(
                *command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(render_pass)
                    .framebuffer(*framebuffer)
                    .render_area(vk::Rect2D::default().extent(swapchain_extent))
                    .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            );
            device.cmd_draw(*command_buffer, 3, 1, 0, 0);
            device.cmd_end_render_pass(*command_buffer);
            device
                .end_command_buffer(*command_buffer)
                .context("Failed to end command buffer.")?;
        }
        Ok(())
    }

    fn create_synchronization(
        device: &Device,
        amount: usize,
    ) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>)>
    {
        let semaphore_builder = vk::SemaphoreCreateInfo::default();
        let fence_builder = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let image_available_semaphores = (0..amount)
            .map(|_| unsafe {
                device
                    .create_semaphore(&semaphore_builder, None)
                    .context("Failed to create an image available semaphore.")
            })
            .collect::<Result<Vec<_>>>()?;

        let queue_submit_complete_semaphores = (0..amount)
            .map(|_| unsafe {
                device
                    .create_semaphore(&semaphore_builder, None)
                    .context("Failed to create a queue submit complete semaphore.")
            })
            .collect::<Result<Vec<_>>>()?;

        let queue_submit_complete_fences = (0..amount)
            .map(|_| unsafe {
                device
                    .create_fence(&fence_builder, None)
                    .context("Failed to create a queue submit complete fence.")
            })
            .collect::<Result<Vec<_>>>()?;

        Ok((
            image_available_semaphores,
            queue_submit_complete_semaphores,
            queue_submit_complete_fences,
        ))
    }

    pub fn init_window(event_loop: &EventLoop<()>) -> window::Window
    {
        let window = window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::PhysicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            //.with_fullscreen(Some(window::Fullscreen::Borderless(None)))
            .build(event_loop)
            .expect("Couldn't create window.");

        window
    }

    unsafe fn draw(&self, current_frame: usize) -> Result<()>
    {
        self.device
            .wait_for_fences(&[self.queue_submit_complete_fences[current_frame]], true, u64::MAX)
            .context("Failed to wait for fence while drawing image.")?;
        self.device
            .reset_fences(&[self.queue_submit_complete_fences[current_frame]])
            .context("Failed to reset fence while drawing image.")?;

        let (image_index, _) = self.swapchain_loader
            .acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[current_frame],
                vk::Fence::null(),
            )
            .context("Failed to acquire next image while drawing.")?;

        let wait_semaphores = [self.image_available_semaphores[current_frame]];
        let wait_dst_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers[image_index as usize]];
        let signal_semaphores = [self.queue_submit_complete_semaphores[current_frame]];

        let submit_infos = [vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)];

        self.device
            .queue_submit(self.queues.graphics_queue, &submit_infos, self.queue_submit_complete_fences[current_frame])
            .context("Error while submitting command buffer to he queue during rendering.")?;

        let wait_semaphores = [self.queue_submit_complete_semaphores[current_frame]];
        let swapchains = [self.swapchain];
        let image_indices = [image_index];

        self.swapchain_loader
            .queue_present(
                self.queues.presentation_queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&wait_semaphores)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices),
            )
            .context("Error while presenting image to the swapchain.")?;
        Ok(())
    }

    unsafe fn cleanup_swapchain(&mut self)
    {
        trace!("Cleaning up swapchain");
        for framebuffer in &self.framebuffers {
            self.device.destroy_framebuffer(*framebuffer, None);
        }
        self.device.free_command_buffers(self.command_pool, self.command_buffers.as_slice());
        self.device.destroy_pipeline(self.pipeline, None);
        self.device.destroy_render_pass(self.render_pass, None);
        self.device.destroy_pipeline_layout(self.pipeline_layout, None);
        for swapchain_image in &self.swapchain_images {
            self.device.destroy_image_view(swapchain_image.image_view, None);
        }
        self.swapchain_loader.destroy_swapchain(self.swapchain, None);
    }

    unsafe fn recreate_swapchain(&mut self) -> Result<()>
    {
        self.device.device_wait_idle().unwrap();

        self.cleanup_swapchain();

        trace!("Getting new surface info");
        self.surface_info = SurfaceInfo::get_surface_info(
            &self.surface_loader,
            &self.physical_device,
            &self.surface
        )?;

        trace!("Creating new swapchain extent");
        self.swapchain_extent = self.surface_info.surface_capabilities.current_extent;

        trace!("Creating swapchain");
        self.swapchain = DoomApp::create_swapchain(
            &self.swapchain_loader,
            &self.surface,
            &self.surface_info,
            &self.queue_family_indices,
            &self.swapchain_extent
        )?;

        trace!("Creating swapchain images");
        self.swapchain_images = DoomApp::get_swapchain_images(
            &self.swapchain_loader,
            &self.swapchain,
            &self.surface_info.choose_best_color_format()?.format,
            &self.device
        )?;

        trace!("Creating pipeline layout");
        self.pipeline_layout = Self::create_pipeline_layout(&self.device)?;

        trace!("Creating render pass");
        self.render_pass = Self::create_render_passe(&self.device, &self.surface_info)?;

        self.swapchain_extent = self.surface_info.surface_capabilities.current_extent;

        trace!("Creating pipeline");
        self.pipeline = Self::create_graphics_pipeline(
            *self.shader_modules.get("triangle.vert.spv").unwrap(),
            *self.shader_modules.get("triangle.frag.spv").unwrap(),
            self.swapchain_extent,
            self.pipeline_layout,
            self.render_pass,
            &self.device
        )?;

        trace!("Creating framebuffers");
        self.framebuffers = self.swapchain_images
            .iter()
            .map(|image| {
                Self::create_framebuffer(
                    &self.device,
                    self.render_pass,
                    image.image_view,
                    self.swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        trace!("Creating command buffers");
        self.command_buffers = Self::allocate_command_buffers(
            &self.device,
            self.command_pool,
            self.framebuffers.len() as u32
        )?;

        trace!("Recording command buffers");
        Self::record_command_buffers(
            &self.device,
            self.command_buffers.as_slice(),
            self.framebuffers.as_slice(),
            self.render_pass,
            self.swapchain_extent,
            self.pipeline
        )?;

        Ok(())
    }

    fn create_buffer(
        &self,
        vertices: &[Vertex]
    ) -> Result<()> {
        let buffer = unsafe {
            self.device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size((mem::size_of::<Vertex>()* vertices.len()) as u64)
                        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None
                )?
        };

        let memory_requirements = unsafe {
            self.device.get_buffer_memory_requirements(buffer)
        };

        let memory_properties = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        };

        let property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let buffer_memory = unsafe {
            self.device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(memory_requirements.size)
                        .memory_type_index(DoomApp::find_valid_memory_type_index(
                                memory_properties,
                                memory_requirements,
                                property_flags
                            ).expect("No valid memory type index found") as u32),
                    None)
        };
        todo!()
    }

    fn find_valid_memory_type_index(
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        memory_requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Option<usize> {
        memory_properties
            .memory_types
            .into_iter()
            .enumerate()
            .position(|(index, memory_type)| {
                (memory_requirements.memory_type_bits & (1 << index as u32)) != 0
                    && memory_type.property_flags.contains(flags)
            })
    }

    pub fn main_loop(mut self, event_loop: EventLoop<()>)
    {
        let mut current_frame = 0;
        let max_frames = self.image_available_semaphores.len();

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
                        },
                        WindowEvent::Resized(..) => unsafe {
                            self.recreate_swapchain().unwrap();
                        }
                        WindowEvent::RedrawRequested => {
                            unsafe {
                                while let Err(..) = self.draw(current_frame) {
                                    self.recreate_swapchain().unwrap();
                                }
                            };
                            current_frame = (current_frame + 1) % max_frames;},
                        _ => (),
                    };
                }
            })
            .unwrap()
    }
}

impl Drop for DoomApp
{
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            for semaphore in &self.image_available_semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }
            for semaphore in &self.queue_submit_complete_semaphores {
                self.device.destroy_semaphore(*semaphore, None);
            }
            for fence in &self.queue_submit_complete_fences {
                self.device.destroy_fence(*fence, None);
            }
            for shader_module in self.shader_modules.values() {
                self.device.destroy_shader_module(*shader_module, None);
            }
            self.cleanup_swapchain();
            self.device.destroy_command_pool(self.command_pool, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            if let Some((utils, messenger)) = self.debug_report_callback.take() {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
