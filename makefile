MAKEFLAGS += --no-print-directory

all: wukong

wukong:
	@cmake -B build && cmake --build build -t wukong

test:
	@cmake -B build -DCF_TEST=on && cmake --build build -t test_cuda # -t test -- ARGS="-V"
	@./build/test_cuda
	@make wukong
	@go test -v ./... -coverprofile=c.out
	@go tool cover -html=c.out -o coverage.html

# run all the benchmarks
bench:
	@cmake -B build && cmake --build build -t wukong
	@go test -run=^$$ -bench=. -benchmem ./...
# for specific benchmark: `make bench-foo`
bench-%:
	@cmake -B build && cmake --build build -t wukong
	@go test -run=^$$ -bench=$* -benchmem ./...

clean:
	@rm -rf build
	@rm coverage.html
	@echo "clean done"
