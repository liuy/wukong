MAKEFLAGS += --no-print-directory

all: wukong

wukong:
	@cmake -B build && cmake --build build -t wukong
	@echo "\n================program output=================\n"
	@./build/wukong

test:
	@cmake -B build -DCF_TEST=on && cmake --build build -t test_cuda # -t test -- ARGS="-V"
	@./build/test_cuda

clean:
	@rm -rf build
	@echo "clean done"