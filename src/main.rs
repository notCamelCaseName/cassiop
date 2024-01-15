mod utility;
mod app_ash;

fn main() {
    let app = app_ash::DoomApp::new();

    app.run();
}
