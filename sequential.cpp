#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <chrono>

struct RGB {
    unsigned char r, g, b;
    RGB() : r(0), g(0), b(0) {}
    RGB(unsigned char red, unsigned char green, unsigned char blue) : r(red), g(green), b(blue) {}
};

class SimpleImage {
private:
    std::vector<std::vector<RGB>> pixels;
    int width, height;

public:
    SimpleImage(int w, int h) : width(w), height(h) {
        pixels.resize(height, std::vector<RGB>(width));
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
        pixels.resize(height, std::vector<RGB>(width));

        if (format == "P3") {
            // ASCII format
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int r, g, b;
                    if (!(file >> r >> g >> b)) {
                        std::cerr << "Error: Failed to read pixel data" << std::endl;
                        return false;
                    }
                    pixels[i][j] = RGB(
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
                    pixels[i][j] = RGB(
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

    RGB getPixel(int x, int y) const {
        // Clamp coordinates to image boundaries
        x = std::max(0, std::min(x, width - 1));
        y = std::max(0, std::min(y, height - 1));
        return pixels[y][x];
    }

    void setPixel(int x, int y, const RGB& color) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            pixels[y][x] = color;
        }
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
};

class GaussianFilter {
private:
    // 5x5 Gaussian kernel (approximation)
    // This is a normalized 5x5 Gaussian filter with sigma ≈ 1.0
    float kernel[5][5] = {
        {1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256},
        {4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256},
        {6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256},
        {4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256},
        {1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256}
    };

public:
    SimpleImage convolve(const SimpleImage& input) {
        int width = input.getWidth();
        int height = input.getHeight();
        SimpleImage output(width, height);

        // Perform convolution on each RGB channel separately
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
                
                // Apply the 5x5 kernel
                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {
                        int px = x + kx;
                        int py = y + ky;
                        
                        // Get pixel value (with boundary handling)
                        RGB pixelValue = input.getPixel(px, py);
                        
                        // Apply kernel weight to each channel
                        float kernelWeight = kernel[ky + 2][kx + 2];
                        sumR += pixelValue.r * kernelWeight;
                        sumG += pixelValue.g * kernelWeight;
                        sumB += pixelValue.b * kernelWeight;
                    }
                }

                // Clamp results to valid range
                sumR = std::max(0.0f, std::min(255.0f, sumR));
                sumG = std::max(0.0f, std::min(255.0f, sumG));
                sumB = std::max(0.0f, std::min(255.0f, sumB));
                
                RGB resultColor(
                    static_cast<unsigned char>(sumR),
                    static_cast<unsigned char>(sumG),
                    static_cast<unsigned char>(sumB)
                );
                output.setPixel(x, y, resultColor);
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

double run_g(const std::string inputJpg) {
    // std::cout << "5x5 Gaussian Filter Convolution (Color Preserving)\n";
    // std::cout << "==================================================\n\n";

    const std::string outputJpg = "results/seq_filtered_image.jpg";
    const std::string tempPpm = "temp_input.ppm";
    const std::string tempFilteredPpm = "temp_filtered.ppm";

    // std::cout << "Converting " << inputJpg << " to PPM format...\n";
    if (!SimpleImage::convertJpgToPpm(inputJpg, tempPpm)) {
        std::cerr << "Error: Failed to convert JPG to PPM. Make sure ImageMagick is installed and " 
                  << inputJpg << " exists.\n";
        std::cerr << "Install ImageMagick with: sudo apt-get install imagemagick (Linux) or brew install imagemagick (Mac)\n";
        return 1;
    }

    // Load the converted PPM image
    // std::cout << "Loading image...\n";
    SimpleImage image(100, 100); // Initial size, will be updated during load
    if (!image.loadPPM(tempPpm)) {
        std::cerr << "Error: Failed to load converted image.\n";
        // Clean up temp file
        remove(tempPpm.c_str());
        return 1;
    }

    // std::cout << "Image loaded: " << image.getWidth() << "x" << image.getHeight() << " pixels\n";

    // Create Gaussian filter and display kernel
    GaussianFilter filter;
    // filter.printKernel();

    // Apply convolution
    // std::cout << "Applying 5x5 Gaussian filter to RGB channels...\n";
    auto start = std::chrono::high_resolution_clock::now();
    SimpleImage filtered = filter.convolve(image);
    auto end = std::chrono::high_resolution_clock::now();

    // Save filtered result as PPM first
    // std::cout << "Saving filtered result...\n";
    if (!filtered.savePPM(tempFilteredPpm)) {
        std::cerr << "Error: Failed to save filtered PPM image.\n";
        remove(tempPpm.c_str());
        return 1;
    }

    // Convert filtered PPM back to JPG
    // std::cout << "Converting filtered image to JPG format...\n";
    if (!SimpleImage::convertPpmToJpg(tempFilteredPpm, outputJpg)) {
        std::cerr << "Error: Failed to convert filtered image to JPG.\n";
        remove(tempPpm.c_str());
        remove(tempFilteredPpm.c_str());
        return 1;
    }

    // Clean up temporary files
    remove(tempPpm.c_str());
    remove(tempFilteredPpm.c_str());

    // std::cout << "Convolution complete!\n";
    // std::cout << "Input: " << inputJpg << " (original colors preserved)\n";
    // std::cout << "Output: " << outputJpg << " (filtered with 5x5 Gaussian kernel)\n\n";

    auto duration = std::chrono::duration<double, std::milli>(end - start);
    // std::cout << "Time taken: " << duration.count() << " ms\n";

    return duration.count();
}