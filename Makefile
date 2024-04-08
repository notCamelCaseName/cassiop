export WAYLAND_DISPLAY=""

.PHONY: shaders

main: src/*
	cargo build

main-release: src/*
	cargo build --release

shaders: shaders/*
	for file in $^; do \
		glslc $${file} -o $${file}.spv; \
	done

run: main shaders
	cargo run

release: main-release shaders
	cargo run --release

debug: main shaders
	RUST_LOG=debug cargo run

clean: shaders/*.spv
	cargo clean
	rm shaders/*.spv
