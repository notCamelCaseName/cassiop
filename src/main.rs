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

    let cube = [
        Vertex {
            position: [0.5, 0.5, 0.5],
            color: [0.0, 0.0, 0.0],
        },
        Vertex {
            position: [-0.5, 0.5, 0.5],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-0.5, -0.5, 0.5],
            color: [1.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, -0.5, 0.5],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.5, 0.5, -0.5],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [-0.5, 0.5, -0.5],
            color: [1.0, 0.0, 1.0],
        },
        Vertex {
            position: [-0.5, -0.5, -0.5],
            color: [1.0, 1.0, 1.0],
        },
        Vertex {
            position: [0.5, -0.5, -0.5],
            color: [0.0, 1.0, 1.0],
        },
    ];

    app.load_vertices(
        &cube,
        &[
            0, 1, 2, 0, 2, 3,
            0, 7, 4, 0, 3, 7,
            0, 5, 1, 0, 4, 5,
            6, 3, 2, 6, 7, 3,
            6, 5, 4, 6, 4, 7,
            6, 1, 5, 6, 2, 1,
        ]
    ).unwrap();


    info!("Running");
    app.run(event_loop, window);
    Ok(())
}
