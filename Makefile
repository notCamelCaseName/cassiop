export WAYLAND_DISPLAY=""

main: src/*
	cargo build

run: main shaders/*
	cargo run

debug: main shaders/*
	RUST_LOG=debug cargo run
