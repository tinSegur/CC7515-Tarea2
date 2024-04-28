BUILD=build
PROJECT=MyProject

.PHONY: build init clean test run

init:
	cmake -S . -B $(BUILD)

build:
	cmake --build $(BUILD) -j 10

run:
	@./$(BUILD)/src/$(PROJECT)Run

test:
	ctest --test-dir $(BUILD) --output-on-failure

clean:
	rm -rf .cache Testing build

watch:
	find src/ test/ | entr -s "make build"
