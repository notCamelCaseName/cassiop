mod utility;
mod debug;
mod app_ash;

use log::info;

fn main() {
    env_logger::init();
    let app = app_ash::DoomApp::new();

    info!("Running");

    app.run();
}
