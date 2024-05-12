pub use std::collections::HashMap;
use std::fmt::Error;
pub use std::mem;
use anyhow::anyhow;
pub use ash::Device;
pub use ash::prelude::VkResult;
pub use winit::window::Window;
pub(crate) use {
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
pub use crate::app_ash::Vertex;
pub use crate::utility::create_shader_module;

pub const WINDOW_TITLE: &str = "DoomApp";
pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 600;

pub const MAX_FRAMES: usize = 2;

pub const FILL_COLOR: [vk::ClearValue; 1] = [vk::ClearValue {
    color: vk::ClearColorValue {
        float32: [0.1, 0.1, 0.1 , 1.0],
    },
}];

pub struct Queues {
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) presentation_queue: vk::Queue
}

pub struct QueueFamilyIndices {
    pub(crate) graphics_family: Option<u32>,
    presentation_family: Option<u32>
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.presentation_family.is_some()
    }
}

pub struct SwapchainImage {
    pub(crate) image: vk::Image,
    pub(crate) image_view: vk::ImageView
}
pub fn create_instance(entry: Arc<ash::Entry>, window: &Window) -> Result<ash::Instance>
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

    unsafe { validate_required_extensions(reqs, entry.clone())? };

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

pub unsafe fn validate_required_extensions(
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

pub fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice>
{
    unsafe {
        Ok(instance
            .enumerate_physical_devices()?
            .iter()
            .find(|&dev| {
                trace!(
                        "Found physical device : {}",
                        utility::mnt_to_string(
                            &instance.get_physical_device_properties(*dev).device_name
                        )
                    );
                check_device_extension_support(instance, dev)
            })
            .expect("No physical devices supporting required extensions found")
            .to_owned())
    }
}

pub fn get_queue_family_indices(
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

pub fn create_logical_device(
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

pub fn check_device_extension_support(
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

pub fn create_swapchain(
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

pub fn create_image_view(
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

pub fn get_swapchain_images(
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
            let image_view = create_image_view(&image, format, vk::ImageAspectFlags::COLOR, device)?;
            Ok::<SwapchainImage, anyhow::Error>(SwapchainImage {image, image_view})
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(swapchain_images_output)
}

pub fn create_pipeline_layout(
    device: &Device,
    set_layouts: &[vk::DescriptorSetLayout],
) -> Result<vk::PipelineLayout>
{
    unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(set_layouts)
                .push_constant_ranges(&[]),
            None,
        )
    }.context("Error trying to create a pipeline layout.")
}

pub fn create_render_passe(
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

pub fn create_graphics_pipeline(
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

    let attribute_descriptions = [
        vk::VertexInputAttributeDescription::default()
            .format(vk::Format::R32G32B32_SFLOAT),
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

pub fn create_framebuffer(
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

pub fn create_command_pool(device: &Device, queue_family_index: u32) -> Result<vk::CommandPool>
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

pub fn allocate_command_buffers(
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

pub fn create_synchronization(
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

pub fn find_valid_memory_type_index(
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    memory_requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Option<usize>
{
    memory_properties
        .memory_types
        .into_iter()
        .enumerate()
        .position(|(index, memory_type)| {
            (memory_requirements.memory_type_bits & (1 << index as u32)) != 0
                && memory_type.property_flags.contains(flags)
        })
}

pub fn create_buffer(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    usage: vk::BufferUsageFlags,
    memory_property_flags: vk::MemoryPropertyFlags,
    size: vk::DeviceSize,
) -> Result<(vk::Buffer, vk::DeviceMemory)>
{
    unsafe {
        // create a buffer handle of the right size and type.
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;

        // get buffer memory requirements plus the memory properties of our physical device.
        let memory_requirements = device.get_buffer_memory_requirements(buffer);
        let memory_properties = instance.get_physical_device_memory_properties(physical_device);

        // find a valid memory type index to use.
        let memory_type_index = find_valid_memory_type_index(
            memory_properties,
            memory_requirements,
            memory_property_flags,
        )
            .ok_or_else(|| anyhow!("Failed to get a valid memory type for buffer."))?;

        // allocate memory.
        let buffer_memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(memory_type_index as u32),
                None,
            )
            .context("Failed to allocate buffer memory.")?;

        // bind buffer memory.
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .context("Failed to bind buffer memory to the buffer.")?;

        // return.
        Ok::<(vk::Buffer, vk::DeviceMemory), Error>((buffer, buffer_memory))
    }
        .context("Error when trying to create a buffer of some type.")
}

pub fn create_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout>
{
    unsafe {
        device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    vk::DescriptorSetLayoutBinding::default()
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::VERTEX),
                ]),
                None,
            )
            .context("Failed to create a descriptor set layout.")
    }
}

pub fn create_uniform_buffers(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    count: usize,
) -> Result<Vec<(vk::Buffer, vk::DeviceMemory)>>
{
    let mut buffers = Vec::with_capacity(count);
    for _ in 0..count {
        buffers.push(
            create_buffer(
                instance,
                device,
                physical_device,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                mem::size_of::<crate::app_ash::ModelViewProjection>().try_into().unwrap(),
            )
                .context("Failed to create a uniform buffer.")?,
        );
    }
    Ok(buffers)
}

pub fn create_descriptor_pool(device: &Device, count: u32) -> Result<vk::DescriptorPool>
{
    unsafe {
        device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(count)
                    .pool_sizes(&[vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(count)]),
                None,
            )
            .context("Failed to create a descriptor pool.")
    }
}

pub fn allocate_descriptor_sets(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    count: usize,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = std::iter::repeat(descriptor_set_layout)
        .take(count)
        .collect::<Vec<_>>();
    unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )
            .context("Failed to allocate descriptor sets.")
    }
}