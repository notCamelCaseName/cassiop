.PHONY: shaders

main: src/*
	cargo build

main-release: src/*
	cargo build --release

shaders: frag-shaders vert-shaders

frag-shaders: shaders/*.frag
	for file in $^; do \
		glslc $${file} -o $${file}.spv; \
	done
vert-shaders: shaders/*.vert
	for file in $^; do \
		glslc $${file} -o $${file}.spv; \
	done

run: shaders main
	WAYLAND_DISPLAY="" cargo run

release: shaders main-release
	WAYLAND_DISPLAY="" cargo run --release

debug: shaders main
	WAYLAND_DISPLAY="" RUST_LOG=cassiop=trace cargo run

clean: shaders/*.spv
	cargo clean
	rm shaders/*.spv
