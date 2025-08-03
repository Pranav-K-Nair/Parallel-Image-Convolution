#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <memory>
#include <immintrin.h>  // For AVX
#include <algorithm>

struct RGB2 {
    unsigned char r, g, b;
    RGB2() : r(0), g(0), b(0) {}
    RGB2(unsigned char red, unsigned char green, unsigned char blue) : r(red), g(green), b(blue) {}
};

class SimpleImage2 {
private:
    std::vector<std::vector<RGB2>> pixels;
    int width, height;

public:
    SimpleImage2(int w, int h) : width(w), height(h) {
        pixels.resize(height, std::vector<RGB2>(width));
    }

    // Load a PPM (Portable PixMap) format image - RGB color
    bool loadPPM(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return false;
        }

        std::string format;
        file >> format;
        if (format != "P3" && format != "P6") {
            std::cerr << "Error: Only P3 and P6 PPM formats supported" << std::endl;
            return false;
        }

        // Skip comments
        std::string line;
        while (file.peek() == '#') {
            std::getline(file, line);
        }

        int maxVal;
        file >> width >> height >> maxVal;
        
        if (width <= 0 || height <= 0) {
            std::cerr << "Error: Invalid image dimensions" << std::endl;
            return false;
        }
        
        // Clear existing data and resize
        pixels.clear();
        pixels.resize(height, std::vector<RGB2>(width));

        if (format == "P3") {
            // ASCII format
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int r, g, b;
                    if (!(file >> r >> g >> b)) {
                        std::cerr << "Error: Failed to read pixel data" << std::endl;
                        return false;
                    }
                    pixels[i][j] = RGB2(
                        static_cast<unsigned char>(std::max(0, std::min(255, r))),
                        static_cast<unsigned char>(std::max(0, std::min(255, g))),
                        static_cast<unsigned char>(std::max(0, std::min(255, b)))
                    );
                }
            }
        } else {
            // Binary format P6
            file.ignore(); // Skip whitespace after header
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int r = file.get();
                    int g = file.get();
                    int b = file.get();
                    if (r == EOF || g == EOF || b == EOF) {
                        std::cerr << "Error: Unexpected end of file" << std::endl;
                        return false;
                    }
                    pixels[i][j] = RGB2(
                        static_cast<unsigned char>(r),
                        static_cast<unsigned char>(g),
                        static_cast<unsigned char>(b)
                    );
                }
            }
        }
        file.close();
        return true;
    }

    // Save as PPM format (color)
    bool savePPM(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot create file " << filename << std::endl;
            return false;
        }

        file << "P3\n";
        file << width << " " << height << "\n";
        file << "255\n";

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                file << static_cast<int>(pixels[i][j].r) << " "
                     << static_cast<int>(pixels[i][j].g) << " "
                     << static_cast<int>(pixels[i][j].b) << " ";
            }
            file << "\n";
        }
        file.close();
        return true;
    }

    // Convert JPG to PPM using ImageMagick (color format)
    static bool convertJpgToPpm(const std::string& jpgFile, const std::string& ppmFile) {
        std::string command = "convert \"" + jpgFile + "\" \"" + ppmFile + "\"";
        int result = system(command.c_str());
        return result == 0;
    }
    
    // Convert PPM to JPG using ImageMagick
    static bool convertPpmToJpg(const std::string& ppmFile, const std::string& jpgFile) {
        std::string command = "convert \"" + ppmFile + "\" \"" + jpgFile + "\"";
        int result = system(command.c_str());
        return result == 0;
    }

    RGB2 getPixel(int x, int y) const {
        // Clamp coordinates to image boundaries
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        return pixels[y][x];
    }

    void setPixel(int x, int y, const RGB2& color) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            pixels[y][x] = color;
        }
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
    // Get raw pixel data for SIMD operations
    const std::vector<std::vector<RGB2>>& getPixelData() const { return pixels; }
    std::vector<std::vector<RGB2>>& getPixelData() { return pixels; }
};

class SIMDGaussianFilter2 {
private:
    // 5x5 Gaussian kernel (approximation)
    alignas(32) float kernel[5][5] = {
        {1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256},
        {4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256},
        {6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256},
        {4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256},
        {1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256}
    };

    // Adaptive tile size based on image size and available cache
    static int getTileSize(int width, int height) {
        // For small images, don't use tiling (overhead not worth it)
        if (width * height < 50000) return std::max(width, height);
        
        // For medium images, use smaller tiles
        if (width * height < 500000) return 32;
        
        // For large images, use larger tiles
        return 64;
    }

    // Check for AVX2 support at runtime
    bool checkAVX2Support() const {
        return __builtin_cpu_supports("avx2");
    }

    // Improved SIMD-optimized convolution using AVX2 with better memory access patterns
    void convolveRowAVX2(const SimpleImage2& input, SimpleImage2& output, int y, int startX, int endX) const {
        const auto& inputPixels = input.getPixelData();
        auto& outputPixels = output.getPixelData();
        int width = input.getWidth();
        int height = input.getHeight();

        // Only use SIMD for sufficiently wide regions (overhead vs benefit trade-off)
        if (endX - startX < 16) {
            convolveRowScalar(input, output, y, startX, endX);
            return;
        }

        // Pre-allocate aligned buffers to reduce memory allocation overhead
        alignas(32) static thread_local float pixelsR[8], pixelsG[8], pixelsB[8];
        alignas(32) static thread_local float resultR[8], resultG[8], resultB[8];

        // Process 8 pixels at a time with AVX2
        for (int x = startX; x < endX - 7; x += 8) {
            __m256 sumR = _mm256_setzero_ps();
            __m256 sumG = _mm256_setzero_ps();
            __m256 sumB = _mm256_setzero_ps();

            // Apply 5x5 kernel - process kernel weights in groups to improve cache usage
            for (int ky = -2; ky <= 2; ky++) {
                int py = std::max(0, std::min(y + ky, height - 1));
                
                // Pre-fetch the entire row section we'll need
                for (int kx = -2; kx <= 2; kx++) {
                    float kernelWeight = kernel[ky + 2][kx + 2];
                    __m256 weight = _mm256_set1_ps(kernelWeight);

                    // Improved pixel loading with better memory access pattern
                    // Check if we can do contiguous loading (no boundary clamping needed)
                    if (x + kx >= 2 && x + 7 + kx < width - 2) {
                        // Fast path: contiguous memory access
                        const RGB2* rowPtr = &inputPixels[py][x + kx];
                        for (int i = 0; i < 8; i++) {
                            pixelsR[i] = static_cast<float>(rowPtr[i].r);
                            pixelsG[i] = static_cast<float>(rowPtr[i].g);
                            pixelsB[i] = static_cast<float>(rowPtr[i].b);
                        }
                    } else {
                        // Slow path: boundary clamping required
                        for (int i = 0; i < 8; i++) {
                            int px = std::max(0, std::min(x + i + kx, width - 1));
                            const RGB2& pixel = inputPixels[py][px];
                            pixelsR[i] = static_cast<float>(pixel.r);
                            pixelsG[i] = static_cast<float>(pixel.g);
                            pixelsB[i] = static_cast<float>(pixel.b);
                        }
                    }

                    __m256 vR = _mm256_load_ps(pixelsR);
                    __m256 vG = _mm256_load_ps(pixelsG);
                    __m256 vB = _mm256_load_ps(pixelsB);

                    // Use FMA for better performance and accuracy
                    sumR = _mm256_fmadd_ps(vR, weight, sumR);
                    sumG = _mm256_fmadd_ps(vG, weight, sumG);
                    sumB = _mm256_fmadd_ps(vB, weight, sumB);
                }
            }

            // Store results with clamping
            _mm256_store_ps(resultR, sumR);
            _mm256_store_ps(resultG, sumG);
            _mm256_store_ps(resultB, sumB);

            // Convert to unsigned char with SIMD clamping would be ideal, but for simplicity:
            for (int i = 0; i < 8 && (x + i) < endX; i++) {
                unsigned char r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, resultR[i])));
                unsigned char g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, resultG[i])));
                unsigned char b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, resultB[i])));
                outputPixels[y][x + i] = RGB2(r, g, b);
            }
        }

        // Handle remaining pixels with scalar processing
        int remainingStart = ((endX - startX) / 8) * 8 + startX;
        if (remainingStart < endX) {
            convolveRowScalar(input, output, y, remainingStart, endX);
        }
    }


    // Scalar fallback for systems without SIMD support
    void convolveRowScalar(const SimpleImage2& input, SimpleImage2& output, int y, int startX, int endX) const {
        const auto& inputPixels = input.getPixelData();
        auto& outputPixels = output.getPixelData();
        int width = input.getWidth();
        int height = input.getHeight();

        for (int x = startX; x < endX; x++) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int px = std::max(0, std::min(x + kx, width - 1));
                    int py = std::max(0, std::min(y + ky, height - 1));
                    
                    const RGB2& pixel = inputPixels[py][px];
                    float kernelWeight = kernel[ky + 2][kx + 2];
                    
                    sumR += pixel.r * kernelWeight;
                    sumG += pixel.g * kernelWeight;
                    sumB += pixel.b * kernelWeight;
                }
            }

            unsigned char r = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, sumR)));
            unsigned char g = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, sumG)));
            unsigned char b = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, sumB)));
            outputPixels[y][x] = RGB2(r, g, b);
        }
    }

public:
    // Cache-aware tiled convolution with SIMD optimization
    SimpleImage2 convolve(const SimpleImage2& input) {
        int width = input.getWidth();
        int height = input.getHeight();
        SimpleImage2 output(width, height);

        // Detect available SIMD instruction sets
        bool hasAVX2 = checkAVX2Support();
        
        // std::cout << "SIMD Support detected: ";
        // if (hasAVX2) {
        //     std::cout << "AVX2 (8-wide SIMD)\n";
        // }
        // else {
        //     std::cout << "Scalar only (no SIMD)\n";
        // }

        // Adaptive tile size based on image characteristics
        int tileSize = getTileSize(width, height);
        // std::cout << "Using tile size: " << tileSize << "x" << tileSize << "\n";
        
        // For very small images, just use scalar processing (SIMD overhead not worth it)
        if (width * height < 10000) {
            std::cout << "Small image detected - using scalar processing\n";
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < height; y++) {
                convolveRowScalar(input, output, y, 0, width);
            }
            return output;
        }

        // Adjust number of threads for small images to avoid overhead
        int optimalThreads = std::min(omp_get_max_threads(), std::max(1, height / 4));
        
        // Process image in cache-friendly tiles
        #pragma omp parallel for schedule(dynamic) collapse(2) num_threads(optimalThreads)
        for (int tileY = 0; tileY < height; tileY += tileSize) {
            for (int tileX = 0; tileX < width; tileX += tileSize) {
                int endY = std::min(tileY + tileSize, height);
                int endX = std::min(tileX + tileSize, width);
                
                // Process each row in the tile
                for (int y = tileY; y < endY; y++) {
                    if (hasAVX2 && endX - tileX >= 16) {
                        convolveRowAVX2(input, output, y, tileX, endX);
                    } else {
                        convolveRowScalar(input, output, y, tileX, endX);
                    }
                }
            }
        }

        return output;
    }

    void printKernel() const {
        std::cout << "5x5 Gaussian Kernel:\n";
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                std::cout << kernel[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

double run_ag(const std::string inputJpg) {
    // std::cout << "SIMD Gaussian Filter with Advanced CPU Optimizations\n";
    // std::cout << "====================================================\n\n";

    // Display system information
    // std::cout << "System Information:\n";
    // std::cout << "OpenMP Support: ";
    // #ifdef _OPENMP
    //     std::cout << "Enabled\n";
    //     std::cout << "Max threads available: " << omp_get_max_threads() << "\n";
    //     std::cout << "Number of processors: " << omp_get_num_procs() << "\n";
    // #else
    //     std::cout << "Disabled\n";
    // #endif

    // CPU feature detection
    // std::cout << "CPU Features:\n";
    // std::cout << "AVX2 Support: " << (__builtin_cpu_supports("avx2") ? "Yes" : "No") << "\n";
    // std::cout << "FMA Support: " << (__builtin_cpu_supports("fma") ? "Yes" : "No") << "\n\n";

    // const std::string inputJpg = "images/image.jpg";
    const std::string outputJpg = "results/advanced_cpu_filtered_image.jpg";
    const std::string tempPpm = "temp_input.ppm";
    const std::string tempFilteredPpm = "temp_filtered.ppm";

    // std::cout << "Converting " << inputJpg << " to PPM format...\n";
    if (!SimpleImage2::convertJpgToPpm(inputJpg, tempPpm)) {
        std::cerr << "Error: Failed to convert JPG to PPM. Make sure ImageMagick is installed and " 
                  << inputJpg << " exists.\n";
        std::cerr << "Install ImageMagick with: sudo apt-get install imagemagick (Linux) or brew install imagemagick (Mac)\n";
        return 1;
    }

    // Load the converted PPM image
    // std::cout << "Loading image...\n";
    SimpleImage2 image(100, 100); // Initial size, will be updated during load
    if (!image.loadPPM(tempPpm)) {
        std::cerr << "Error: Failed to load converted image.\n";
        remove(tempPpm.c_str());
        return 1;
    }

    // std::cout << "Image loaded: " << image.getWidth() << "x" << image.getHeight() << " pixels\n";

    // Create SIMD filter
    SIMDGaussianFilter2 filter;

    // Apply SIMD convolution
    // std::cout << "APPLYING SIMD GAUSSIAN FILTER\n";
    auto start = std::chrono::high_resolution_clock::now();
    SimpleImage2 filtered = filter.convolve(image);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);

    // Save result
    // std::cout << "Saving filtered result...\n";
    if (!filtered.savePPM(tempFilteredPpm)) {
        std::cerr << "Error: Failed to save filtered PPM image.\n";
        remove(tempPpm.c_str());
        return 1;
    }

    // Convert filtered PPM back to JPG
    // std::cout << "Converting filtered image to JPG format...\n";
    if (!SimpleImage2::convertPpmToJpg(tempFilteredPpm, outputJpg)) {
        std::cerr << "Error: Failed to convert filtered image to JPG.\n";
        remove(tempPpm.c_str());
        remove(tempFilteredPpm.c_str());
        return 1;
    }

    // Clean up temporary files
    remove(tempPpm.c_str());
    remove(tempFilteredPpm.c_str());

    // Clean up
    remove(tempPpm.c_str());

    // std::cout << "Convolution complete!\n";
    // std::cout << "Input: " << inputJpg << "\n";
    // std::cout << "Output: " << outputJpg << " (SIMD + Cache Optimized)\n\n";
    // std::cout << "Processing time: " << duration.count() << " ms\n";

    // #ifdef _OPENMP
    //     std::cout << "Used " << omp_get_max_threads() << " OpenMP threads\n";
    // #endif

    return duration.count();
}