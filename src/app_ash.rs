use glam::*;
use crate::vulkan_functions::*;
use ash::vk;

#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3], // offset 0
    pub color: [f32; 3],    // offset 12
}

#[repr(C)]
pub struct ModelViewProjection {
    projection: Mat4,
    view: Mat4,
    model: Mat4,
}

pub struct Mesh {
    pub(crate) vertex_buffer: Buffer,
    pub(crate) vertex_buffer_memory: DeviceMemory,
    pub(crate) index_buffer: Buffer,
    pub(crate) index_buffer_memory: DeviceMemory,
    pub(crate) index_count: usize
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
    set_layouts: Vec<vk::DescriptorSetLayout>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    queue_submit_complete_semaphores: Vec<vk::Semaphore>,
    queue_submit_complete_fences: Vec<vk::Fence>,
    meshes: Vec<Mesh>,
}

impl DoomApp
{
    pub fn new(window: &Window) -> Result<Self>
    {
        debug!("Creating entry");
        let entry = Arc::new(ash::Entry::linked());

        debug!("Creating instance");
        let instance = create_instance(entry.clone(), &window)?;
        debug!("Picking physical device");
        let physical_device = pick_physical_device(&instance)?;
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

        let mut surface_info = SurfaceInfo::get_surface_info(&surface_loader, &physical_device, &surface)?;
        // Fix for wayland
        surface_info.surface_capabilities.current_extent = vk::Extent2D{
            width: window.inner_size().width,
            height: window.inner_size().height
        };

        let queue_family_indices = get_queue_family_indices(&physical_device, &instance, &surface_loader, &surface).expect("No queue family indices found");

        debug!("Creating logical device");
        let (queues, device) = create_logical_device(&instance, &queue_family_indices, &physical_device);

        debug!("Creating swapchain");
        let swapchain_loader = swapchain::Device::new(&instance, &device);
        let swapchain_extent = surface_info.surface_capabilities.current_extent;
        let swapchain = create_swapchain(
            &swapchain_loader,
            &surface,
            &surface_info,
            &queue_family_indices,
            &swapchain_extent
        )?;

        let swapchain_images = get_swapchain_images(&swapchain_loader, &swapchain, &surface_info.choose_best_color_format()?.format, &device)?;

        debug!("Loading shaders");
        let mut shader_modules: HashMap<String, ShaderModule> = HashMap::new();
        for elt in std::fs::read_dir("shaders/")? {
            let elt = elt?;
            if elt.path().extension().unwrap().to_str() == Some("spv") {
                let shader_bin = std::fs::read(elt.path())?;
                shader_modules.insert(elt.file_name().into_string().unwrap(), create_shader_module(&device, &shader_bin));
            }
        }

        let mut set_layouts = Vec::new();
        //let uniform_buffers_descriptor_set_layouts = create_descriptor_set_layout(&device)?;
        //set_layouts.push(uniform_buffers_descriptor_set_layouts);

        let pipeline_layout = create_pipeline_layout(
            &device,
            set_layouts.as_slice(),
        )?;

        debug!("Creating render pass");
        let render_pass = create_render_passe(&device, &surface_info)?;

        debug!("Creating pipeline");
        let pipeline = create_graphics_pipeline(
            *shader_modules.get("triangle.vert.spv").unwrap(),
            *shader_modules.get("triangle.frag.spv").unwrap(),
            swapchain_extent,
            pipeline_layout,
            render_pass,
            &device
        )?;

        debug!("Creating framebuffers");
        let framebuffers = swapchain_images
            .iter()
            .map(|image| {
                create_framebuffer(
                    &device,
                    render_pass,
                    image.image_view,
                    swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        debug!("Creating commands");
        let command_pool = create_command_pool(
            &device,
            queue_family_indices.graphics_family.unwrap()
        )?;

        let command_buffers = allocate_command_buffers(
            &device,
            command_pool,
            framebuffers.len() as u32
        )?;

        debug!("Creating semaphores");
        let (
            image_available_semaphores,
            queue_submit_complete_semaphores,
            queue_submit_complete_fences,
        ) = create_synchronization(&device, MAX_FRAMES)?;

        info!("Initialization done");
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
            set_layouts,
            pipeline_layout,
            pipeline,
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            queue_submit_complete_semaphores,
            queue_submit_complete_fences,
            meshes: Vec::new(),
        })
    }
    unsafe fn record_command_buffers(&self) -> Result<()>
    {
        for (command_buffer, framebuffer) in self.command_buffers.iter().zip(&self.framebuffers) {
            self.device
                .begin_command_buffer(
                    *command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
                )
                .context("Failed to begin command buffer.")?;

            self.device.cmd_begin_render_pass(
                *command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(*framebuffer)
                    .render_area(vk::Rect2D::default().extent(self.swapchain_extent))
                    .clear_values(&FILL_COLOR),
                vk::SubpassContents::INLINE,
            );

            self.device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            for mesh in &self.meshes {
                self.device.cmd_bind_vertex_buffers(*command_buffer, 0, &[mesh.vertex_buffer], &[0]);
                self.device.cmd_bind_index_buffer(
                    *command_buffer,
                    mesh.index_buffer,
                    0,
                    IndexType::UINT16,
                );
                self.device.cmd_draw_indexed(
                    *command_buffer,
                    mesh.index_count.try_into().unwrap(),
                    1,
                    0,
                    0,
                    0,
                );
            }

            self.device.cmd_end_render_pass(*command_buffer);

            self.device
                .end_command_buffer(*command_buffer)
                .context("Failed to end command buffer.")?;
        }
        Ok(())
    }

    pub fn init_window(event_loop: &EventLoop<()>) -> Window
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
            .context("Error while submitting command buffer to the queue during rendering.")?;

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

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()>
    {
        self.device.device_wait_idle().unwrap();

        self.cleanup_swapchain();

        trace!("Getting new surface info");
        self.surface_info = SurfaceInfo::get_surface_info(
            &self.surface_loader,
            &self.physical_device,
            &self.surface
        )?;

        // Fix for wayland
        self.surface_info.surface_capabilities.current_extent = vk::Extent2D{
            width: window.inner_size().width,
            height: window.inner_size().height
        };

        trace!("Creating new swapchain extent");
        self.swapchain_extent = self.surface_info.surface_capabilities.current_extent;

        trace!("Creating swapchain");
        self.swapchain = create_swapchain(
            &self.swapchain_loader,
            &self.surface,
            &self.surface_info,
            &self.queue_family_indices,
            &self.swapchain_extent
        )?;

        trace!("Creating swapchain images");
        self.swapchain_images = get_swapchain_images(
            &self.swapchain_loader,
            &self.swapchain,
            &self.surface_info.choose_best_color_format()?.format,
            &self.device
        )?;

        trace!("Creating pipeline layout");
        self.pipeline_layout = create_pipeline_layout(
            &self.device,
            self.set_layouts.as_slice()
        )?;

        trace!("Creating render pass");
        self.render_pass = create_render_passe(&self.device, &self.surface_info)?;

        self.swapchain_extent = self.surface_info.surface_capabilities.current_extent;

        trace!("Creating pipeline");
        self.pipeline = create_graphics_pipeline(
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
                create_framebuffer(
                    &self.device,
                    self.render_pass,
                    image.image_view,
                    self.swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        trace!("Creating command buffers");
        self.command_buffers = allocate_command_buffers(
            &self.device,
            self.command_pool,
            self.framebuffers.len() as u32
        )?;

        trace!("Recording command buffers");
        self.record_command_buffers()?;

        Ok(())
    }

    pub fn load_vertices(
        &mut self,
        vertices: &[Vertex],
        indices: &[u16]
    ) -> Result<()>
    {
        self.meshes.push(unsafe {create_mesh(
            &self.instance,
            &self.device,
            self.physical_device,
            self.command_pool,
            self.queues.presentation_queue,
            vertices,
            indices
        )?});
        Ok(())
    }

    pub fn run(mut self, event_loop: EventLoop<()>, window: Window)
    {
        let mut current_frame = 0;
        let max_frames = self.image_available_semaphores.len();

        let mut timestamp = std::time::Instant::now();

        unsafe {self.record_command_buffers().unwrap()};

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
                            self.recreate_swapchain(&window).unwrap();
                        }
                        WindowEvent::RedrawRequested => {
                            unsafe {
                                while let Err(..) = self.draw(current_frame) {
                                    debug!("Couldn't draw, recreating swapchain");
                                    self.recreate_swapchain(&window).unwrap();
                                    debug!("OK")
                                }
                            };
                            current_frame = (current_frame + 1) % max_frames;
                            let new_timestamp = std::time::Instant::now();
                            let elapsed = new_timestamp - timestamp;
                            let fps = 1./elapsed.as_secs_f64();
                            trace!("fps: {}", fps as u16);
                            timestamp = new_timestamp;
                            window.request_redraw();
                        },
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

            for mesh in &self.meshes {
                self.device.free_memory(mesh.index_buffer_memory, None);
                self.device.destroy_buffer(mesh.index_buffer, None);
                self.device.free_memory(mesh.vertex_buffer_memory, None);
                self.device.destroy_buffer(mesh.vertex_buffer, None);
            }
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
