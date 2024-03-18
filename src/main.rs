mod utility;
mod app_ash;

use log::info;

use winit::event_loop::EventLoop;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = app_ash::DoomApp::init_window(&event_loop);
    let app = app_ash::DoomApp::new(&window);

    info!("Running");
    app.main_loop(event_loop);
}
