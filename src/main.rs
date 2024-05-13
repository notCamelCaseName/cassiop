mod app_ash;
mod surface_info;
mod utility;
mod debug;
mod vulkan_functions;
mod wad;

use {anyhow::Result, log::info, winit::event_loop::EventLoop};
use app_ash::Vertex;

fn main() -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let window = app_ash::DoomApp::init_window(&event_loop);
    let mut app = app_ash::DoomApp::new(&window)?;

    let rectangle1 = [
        Vertex {
            position: [-0.8, -0.8, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [-0.4, -0.4, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [-0.8, -0.4, 0.0],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [-0.4, -0.8, 0.0],
            color: [1.0, 1.0, 0.0],
        },
    ];

    let rectangle2 = [
        Vertex {
            position: [0.4, 0.4, 0.0],
            color: [1.0, 0.0, 0.0],
        },
        Vertex {
            position: [0.8, 0.8, 0.0],
            color: [0.0, 1.0, 0.0],
        },
        Vertex {
            position: [0.4, 0.8, 0.0],
            color: [0.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.8, 0.4, 0.0],
            color: [1.0, 1.0, 0.0],
        },
    ];

    app.load_vertices(&rectangle1, &[0, 1, 2, 0, 3, 1]).unwrap();
    app.load_vertices(&rectangle2, &[0, 1, 2, 0, 3, 1]).unwrap();


    info!("Running");
    app.run(event_loop, window);
    Ok(())
}
