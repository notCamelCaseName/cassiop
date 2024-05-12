mod app_ash;
mod surface_info;
mod utility;
mod debug;
mod vulkan_functions;

use {anyhow::Result, log::info, winit::event_loop::EventLoop};
use app_ash::Vertex;

fn main() -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let window = app_ash::DoomApp::init_window(&event_loop);
    let mut app = app_ash::DoomApp::new(&window)?;

    let triangle1 = [
        Vertex {
            position: [-0.8, -0.8, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.8, 0.8, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [-0.8, 0.8, 0.0],
            color: [0.0, 0.0, 1.0],
        },
    ];

    let triangle2 = [
        Vertex {
            position: [-0.8, -0.8, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.8, -0.8, 0.0],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.8, 0.8, 0.0],
            color: [0.0, 1.0, 0.0],
        },
    ];

    app.load_vertices(&triangle1).unwrap();
    app.load_vertices(&triangle2).unwrap();


    info!("Running");
    app.run(event_loop, window);
    Ok(())
}
