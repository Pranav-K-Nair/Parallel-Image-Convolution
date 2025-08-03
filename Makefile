# Compiler settings
NVCC = nvcc
CXX = g++

# Get GTK flags (separate compile and link flags)
GTK_CFLAGS = `pkg-config --cflags gtk+-3.0`
GTK_LIBS = `pkg-config --libs gtk+-3.0`

# Compiler flags
NVCC_FLAGS = -O0 -arch=sm_50
CXX_CFLAGS = -O0 -std=c++11 -fopenmp -march=native -mavx2 -mfma $(GTK_CFLAGS)

# Linker flags
LINK_FLAGS = -lgomp $(GTK_LIBS)

# Target executable
TARGET = main_run

# Source files
CUDA_SRC1 = gaussian_global.cu
CUDA_SRC2 = gaussian_shared.cu
CPP_SRC = main.cpp

# Object files
CUDA_OBJ1 = gaussian_global.o
CUDA_OBJ2 = gaussian_shared.o
CPP_OBJ = main.o

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(CUDA_OBJ1) $(CUDA_OBJ2) $(CPP_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ $(LINK_FLAGS) -o $@

# Rule to compile .cu files
$(CUDA_OBJ1): $(CUDA_SRC1)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(CUDA_OBJ2): $(CUDA_SRC2)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Rule to compile .cpp files
$(CPP_OBJ): $(CPP_SRC)
	$(CXX) $(CXX_CFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(CUDA_OBJ1) $(CUDA_OBJ2) $(CPP_OBJ) $(TARGET)

# Run target with example usage
run: $(TARGET)
	@echo "Usage: ./$(TARGET) <input.jpg/ppm> <output.ppm>"
	@echo "Example: ./$(TARGET) input.jpg output.ppm"

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the executable (default)"
	@echo "  clean   - Remove object files and executable"
	@echo "  run     - Show usage information"
	@echo "  help    - Show this help message"

.PHONY: all clean run help