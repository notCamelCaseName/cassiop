mod app_ash;
mod surface_info;
mod utility;
mod debug;

use {anyhow::Result, log::info, winit::event_loop::EventLoop};

fn main() -> Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let window = app_ash::DoomApp::init_window(&event_loop);
    let app = app_ash::DoomApp::new(&window)?;

    info!("Running");
    app.main_loop(event_loop);
    Ok(())
}
