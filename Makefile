export WAYLAND_DISPLAY=""

main: src/*
	cargo build

run: main shaders/*
	cargo run
