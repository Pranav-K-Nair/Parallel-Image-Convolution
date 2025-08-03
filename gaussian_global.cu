// CUDA 5x5 Gaussian Filter Convolution (Global Memory, 1 thread per output)
// Usage: nvcc gaussian_global.cu -o gaussian_global && ./gaussian_global ../images/image.jpg ../images/gaussian_filtered.ppm
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <string>

struct RGB {
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

__device__ RGB getPixel(const RGB* img, int x, int y, int width, int height) {
    // Clamp coordinates to image boundaries
    x = max(0, min(x, width-1));
    y = max(0, min(y, height-1));
    return img[y * width + x];
}

__global__ void gaussian5x5_global(const RGB* input, RGB* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sumR = 0, sumG = 0, sumB = 0;
    for (int ky = -2; ky <= 2; ky++) {
        for (int kx = -2; kx <= 2; kx++) {
            RGB pixel = getPixel(input, x + kx, y + ky, width, height);
            float w = d_kernel[(ky+2)*5 + (kx+2)];
            sumR += pixel.r * w;
            sumG += pixel.g * w;
            sumB += pixel.b * w;
        }
    }
    RGB out;
    out.r = min(255, max(0, int(sumR + 0.5f)));
    out.g = min(255, max(0, int(sumG + 0.5f)));
    out.b = min(255, max(0, int(sumB + 0.5f)));
    output[y * width + x] = out;
}

// --- PPM I/O ---
bool loadPPM(const std::string& filename, std::vector<RGB>& img, int& width, int& height) {
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

bool savePPM(const std::string& filename, const std::vector<RGB>& img, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.data()), width * height * 3);
    return true;
}

bool convertJpgToPpm(const std::string& jpgFile, const std::string& ppmFile) {
    std::string command = "convert \"" + jpgFile + "\" \"" + ppmFile + "\"";
    int result = system(command.c_str());
    return result == 0;
}

bool convertPpmToJpg(const std::string& ppmFile, const std::string& jpgFile) {
    std::string command = "convert \"" + ppmFile + "\" \"" + jpgFile + "\"";
    int result = system(command.c_str());
    return result == 0;
}

extern "C"{
    float run_gg(std::string inputFile) {
        // if (argc != 3) {
        //     std::cerr << "Usage: " << argv[0] << " <input.jpg/ppm> <output.ppm>\n";
        //     std::cerr << "Example: " << argv[0] << " ../images/image.jpg ../images/output.ppm\n";
        //     return 1;
        // }
        // std::string inputFile = argv[1];
        std::string outputFile = "results/advanced_cpu_filtered_image.jpg";
        
        // Check if input is JPG and convert to PPM if needed
        std::string tempPpmFile = inputFile;
        bool needCleanup = false;
        
        if (inputFile.size() > 4 && inputFile.substr(inputFile.size()-4) == ".jpg") {
            // std::cout << "Converting " << inputFile << " to PPM format..." << std::endl;
            tempPpmFile = inputFile.substr(0, inputFile.size()-4) + "_temp.ppm";
            if (!convertJpgToPpm(inputFile, tempPpmFile)) {
                std::cerr << "Failed to convert " << inputFile << " to PPM format.\n";
                return 1;
            }
            needCleanup = true;
        }
        
        int width, height;
        std::vector<RGB> h_input;
        if (!loadPPM(tempPpmFile, h_input, width, height)) {
            std::cerr << "Failed to load input image: " << tempPpmFile << "\n";
            if (needCleanup) {
                std::string cleanupCmd = "rm -f " + tempPpmFile;
                system(cleanupCmd.c_str());
            }
            return 1;
        }
        
        std::vector<RGB> h_output(width * height);
        RGB *d_input, *d_output;
        cudaMalloc(&d_input, width * height * sizeof(RGB));
        cudaMalloc(&d_output, width * height * sizeof(RGB));
        cudaMemcpy(d_input, h_input.data(), width * height * sizeof(RGB), cudaMemcpyHostToDevice);
        dim3 block(16, 16);
        dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        gaussian5x5_global<<<grid, block>>>(d_input, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        // std::cout << "time taken: " << milliseconds << std::endl;
        
        cudaMemcpy(h_output.data(), d_output, width * height * sizeof(RGB), cudaMemcpyDeviceToHost);
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
            if (convertPpmToJpg(outputFile, jpgFile)) {
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
        
        return milliseconds;
    }
}