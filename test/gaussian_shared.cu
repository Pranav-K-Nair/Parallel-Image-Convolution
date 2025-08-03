// CUDA 5x5 Gaussian Filter Convolution (Shared Memory, 1 thread per output)
// Usage: nvcc gaussian_shared.cu -o gaussian_shared && ./gaussian_shared ../images/image.jpg ../images/gaussian_filtered_shared.ppm
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <string>

struct RGB3 {
    unsigned char r, g, b;
};

// 5x5 Gaussian kernel (normalized, sigma ≈ 1.0)
__constant__ float d_kernel[25] = {
    1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256,
    4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
    6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256,
    4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
    1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256
};

__global__ void gaussian5x5_shared(const RGB3* input, RGB3* output, int width, int height) {
    // Shared memory tile dimensions (blockDim + 4 for 5x5 kernel)
    extern __shared__ unsigned char tile[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int tile_width = blockDim.x + 4;
    int tile_height = blockDim.y + 4;
    int shared_offset = tile_width * tile_height;
    
    // Load the entire tile into shared memory
    // Each thread loads multiple pixels to fill the 20x20 tile
    for (int load_y = ty; load_y < tile_height; load_y += blockDim.y) {
        for (int load_x = tx; load_x < tile_width; load_x += blockDim.x) {
            int img_x = blockIdx.x * blockDim.x + load_x - 2;
            int img_y = blockIdx.y * blockDim.y + load_y - 2;
            int shared_pos = load_y * tile_width + load_x;
            
            // Load R, G, B channels
            for (int c = 0; c < 3; ++c) {
                unsigned char val = 0;
                if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                    RGB3 pixel = input[img_y * width + img_x];
                    if (c == 0) val = pixel.r;
                    else if (c == 1) val = pixel.g;
                    else val = pixel.b;
                }
                tile[shared_pos + c * shared_offset] = val;
            }
        }
    }
    
    __syncthreads();
    
    if (x < width && y < height) {
        float sum[3] = {0, 0, 0};
        for (int ky = 0; ky < 5; ky++) {
            for (int kx = 0; kx < 5; kx++) {
                int sx = tx + kx;
                int sy = ty + ky;
                float w = d_kernel[ky * 5 + kx];
                for (int c = 0; c < 3; ++c) {
                    sum[c] += tile[(sy) * tile_width + (sx) + c * shared_offset] * w;
                }
            }
        }
        RGB3 out;
        out.r = min(255, max(0, int(sum[0] + 0.5f)));
        out.g = min(255, max(0, int(sum[1] + 0.5f)));
        out.b = min(255, max(0, int(sum[2] + 0.5f)));
        output[y * width + x] = out;
    }
}

// --- PPM I/O ---
bool loadPPM(const std::string& filename, std::vector<RGB3>& img, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    std::string format;
    file >> format;
    if (format != "P6") return false;
    char c;
    while (file >> std::ws && file.peek() == '#') std::getline(file, format); // skip comments
    file >> width >> height;
    int maxval; file >> maxval;
    file.get(c); // skip single whitespace
    img.resize(width * height);
    file.read(reinterpret_cast<char*>(img.data()), width * height * 3);
    return true;
}

bool savePPM(const std::string& filename, const std::vector<RGB3>& img, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data()), width * height * 3);
    return true;
}

bool convertJpgToPpm1(const std::string& jpgFile, const std::string& ppmFile) {
    std::string command = "convert \"" + jpgFile + "\" \"" + ppmFile + "\"";
    int result = system(command.c_str());
    return result == 0;
}

bool convertPpmToJpg1(const std::string& ppmFile, const std::string& jpgFile) {
    std::string command = "convert \"" + ppmFile + "\" \"" + jpgFile + "\"";
    int result = system(command.c_str());
    return result == 0;
}

int main() {
    // if (argc != 3) {
    //     std::cerr << "Usage: " << argv[0] << " <input.jpg/ppm> <output.ppm>\n";
    //     std::cerr << "Example: " << argv[0] << " ../images/image.jpg ../images/output.ppm\n";
    //     return 1;
    // }
    // std::string inputFile = argv[1];
    const std::string inputFile = "../images/large_image.jpg";
    const std::string outputFile = "../results/sg_filtered_image.jpg";
    
    // Check if input is JPG and convert to PPM if needed
    std::string tempPpmFile = inputFile;
    bool needCleanup = false;
    
    if (inputFile.size() > 4 && inputFile.substr(inputFile.size()-4) == ".jpg") {
        // std::cout << "Converting " << inputFile << " to PPM format..." << std::endl;
        tempPpmFile = inputFile.substr(0, inputFile.size()-4) + "_temp.ppm";
        if (!convertJpgToPpm1(inputFile, tempPpmFile)) {
            std::cerr << "Failed to convert " << inputFile << " to PPM format.\n";
            return 1;
        }
        needCleanup = true;
    }
    
    int width, height;
    std::vector<RGB3> h_input;
    if (!loadPPM(tempPpmFile, h_input, width, height)) {
        std::cerr << "Failed to load input image: " << tempPpmFile << "\n";
        if (needCleanup) {
            std::string cleanupCmd = "rm -f " + tempPpmFile;
            system(cleanupCmd.c_str());
        }
        return 1;
    }
    
    std::vector<RGB3> h_output(width * height);
    RGB3 *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(RGB3));
    cudaMalloc(&d_output, width * height * sizeof(RGB3));
    cudaMemcpy(d_input, h_input.data(), width * height * sizeof(RGB3), cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
    int tile_width = block.x + 4;
    int tile_height = block.y + 4;
    int shared_mem_size = tile_width * tile_height * 3 * sizeof(unsigned char);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gaussian5x5_shared<<<grid, block, shared_mem_size>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "time taken: " << milliseconds << std::endl;

    cudaMemcpy(h_output.data(), d_output, width * height * sizeof(RGB3), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    if (!savePPM(outputFile, h_output, width, height)) {
        std::cerr << "Failed to save output image: " << outputFile << "\n";
        if (needCleanup) {
            std::string cleanupCmd = "rm -f " + tempPpmFile;
            system(cleanupCmd.c_str());
        }
        return 1;
    }
    // std::cout << "Filtered image saved to " << outputFile << "\n";
    
    // Also save as JPG if output is .ppm
    if (outputFile.size() > 4 && outputFile.substr(outputFile.size()-4) == ".ppm") {
        std::string jpgFile = outputFile.substr(0, outputFile.size()-4) + ".jpg";
        if (convertPpmToJpg1(outputFile, jpgFile)) {
            std::cout << "Filtered image also saved to " << jpgFile << "\n";
        } else {
            std::cerr << "Failed to convert " << outputFile << " to JPG.\n";
        }
    }
    
    // Clean up temporary PPM file if we created one
    if (needCleanup) {
        std::string cleanupCmd = "rm -f " + tempPpmFile;
        system(cleanupCmd.c_str());
    }
    
    // return milliseconds;
    return 0;
} 
