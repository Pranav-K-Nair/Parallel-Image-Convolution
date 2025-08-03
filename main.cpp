#include <gtk/gtk.h>
#include <cairo.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "sequential.cpp"
#include "openmp_filtering.cpp"
#include "advanced_cpu.cpp"

extern "C" {
    float run_gg(std::string inputFile);
    float run_sg(std::string inputFile);
}

GtkWidget *label;
GtkWidget *chart1_area;
GtkWidget *chart2_area;

std::string inputJpg;
std::string selectedDevice = "Device 1"; // Default device
std::string selectedImageType = ""; // Track which image type was last selected
bool isFirstSelection = true; // Track if this is the first time selecting

// Data for the charts
struct ChartData {
    std::vector<double> values;
    std::vector<std::string> labels;
    std::string title;
};

ChartData chart1_data = {{0.0, 0.0, 0.0, 0.0, 0.0}, {"Sequential", "OpenMP", "Advanced CPU", "CUDA - Global", "CUDA - Shared"}, "Live Performance (ms)"};
ChartData chart2_data = {{0.0, 0.0, 0.0, 0.0, 0.0}, {"Sequential", "OpenMP", "Advanced CPU", "CUDA - Global", "CUDA - Shared"}, "Average Performance (ms)"};

double e_time_1, e_time_2, e_time_3, e_time_4, e_time_5 = 0.0;

// Device-specific performance data
struct DeviceData {
    std::vector<double> small_image;
    std::vector<double> large_image;
};

// Arthur
DeviceData device1_data = {
    {459.51, 77.95, 29.01, 0.23, 0.25},  // Small image performance
    {2843.94, 559.29, 208.27, 1.34, 1.38} // Large image performance
};

// Kratos
DeviceData device2_data = {
    {779.50, 192.843, 69.0, 0.24, 0.23},  // Small image performance
    {4908.66, 1204.59, 418.62, 1.17, 0.86} // Large image performance
};

// Function to draw bar chart
gboolean draw_chart(GtkWidget *widget, cairo_t *cr, gpointer user_data) {
    ChartData *data = (ChartData*)user_data;
    
    int width = gtk_widget_get_allocated_width(widget);
    int height = gtk_widget_get_allocated_height(widget);
    
    // Clear background
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    cairo_paint(cr);
    
    if (data->values.empty()) return FALSE;
    
    // Chart dimensions
    int margin = 60;
    int chart_width = width - 2 * margin;
    int chart_height = height - 2 * margin - 40; // Extra space for title
    
    // Find max value for scaling
    double max_val = *std::max_element(data->values.begin(), data->values.end());
    if (max_val <= 0) max_val = 1.0; // Avoid division by zero
    
    // Draw title
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_select_font_face(cr, "Arial", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, 14);
    cairo_text_extents_t title_extents;
    cairo_text_extents(cr, data->title.c_str(), &title_extents);
    cairo_move_to(cr, (width - title_extents.width) / 2, 25);
    cairo_show_text(cr, data->title.c_str());
    
    // Draw axes
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    cairo_set_line_width(cr, 2);
    
    // Y-axis
    cairo_move_to(cr, margin, margin + 20);
    cairo_line_to(cr, margin, margin + chart_height + 20);
    
    // X-axis
    cairo_move_to(cr, margin, margin + chart_height + 20);
    cairo_line_to(cr, margin + chart_width, margin + chart_height + 20);
    cairo_stroke(cr);
    
    // Draw bars
    int bar_width = chart_width / (data->values.size() * 1.5); // Reduce spacing factor
    int bar_spacing = chart_width / (data->values.size() * 3); // More controlled spacing
    
    for (size_t i = 0; i < data->values.size(); ++i) {
        double normalized_height = (data->values[i] / max_val) * chart_height;
        
        int x = margin + i * (bar_width + bar_spacing) + bar_spacing;
        int y = margin + 20 + chart_height - normalized_height;
        
        // Set different colors for each bar
        if (i == 0) cairo_set_source_rgb(cr, 0.8, 0.2, 0.2); // Red - Sequential
        else if (i == 1) cairo_set_source_rgb(cr, 0.2, 0.8, 0.2); // Green - OpenMP
        else if (i == 2) cairo_set_source_rgb(cr, 0.2, 0.2, 0.8); // Blue - Advanced CPU
        else if (i == 3) cairo_set_source_rgb(cr, 0.8, 0.6, 0.2); // Orange - GPU CUDA
        else if (i == 4) cairo_set_source_rgb(cr, 0.6, 0.2, 0.8); // Purple - GPU OpenCL
        else cairo_set_source_rgb(cr, 0.6, 0.6, 0.6); // Gray for additional bars
        
        cairo_rectangle(cr, x, y, bar_width, normalized_height);
        cairo_fill(cr);
        
        // Draw value on top of bar
        cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
        cairo_set_font_size(cr, 10);
        std::string value_str = std::to_string(data->values[i]);
        value_str = value_str.substr(0, value_str.find('.') + 3); // Keep 2 decimal places
        
        cairo_text_extents_t text_extents;
        cairo_text_extents(cr, value_str.c_str(), &text_extents);
        cairo_move_to(cr, x + (bar_width - text_extents.width) / 2, y - 5);
        cairo_show_text(cr, value_str.c_str());
        
        // Draw label below bar - centered properly
        cairo_text_extents_t label_extents;
        cairo_text_extents(cr, data->labels[i].c_str(), &label_extents);
        cairo_move_to(cr, x + (bar_width - label_extents.width) / 2, 
                     margin + chart_height + 35);
        cairo_show_text(cr, data->labels[i].c_str());
    }
    
    // Draw Y-axis scale
    cairo_set_font_size(cr, 8);
    for (int i = 0; i <= 5; ++i) {
        double scale_value = (max_val / 5) * i;
        int y_pos = margin + 20 + chart_height - (chart_height / 5) * i;
        
        std::string scale_str = std::to_string(scale_value);
        scale_str = scale_str.substr(0, scale_str.find('.') + 2);
        
        cairo_move_to(cr, margin - 30, y_pos + 3);
        cairo_show_text(cr, scale_str.c_str());
        
        // Draw grid line
        cairo_set_source_rgb(cr, 0.9, 0.9, 0.9);
        cairo_set_line_width(cr, 1);
        cairo_move_to(cr, margin, y_pos);
        cairo_line_to(cr, margin + chart_width, y_pos);
        cairo_stroke(cr);
        cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
    }
    
    return FALSE;
}

void update_chart1() {
    chart1_data.values[0] = e_time_1;
    chart1_data.values[1] = e_time_2;
    chart1_data.values[2] = e_time_3;
    chart1_data.values[3] = e_time_4;
    chart1_data.values[4] = e_time_5;
    gtk_widget_queue_draw(chart1_area);
}

void update_chart2() {
    gtk_widget_queue_draw(chart2_area);
}

// Function to update static chart based on current device and image type
void update_static_chart() {
    if (selectedImageType.empty()) return; // No image type selected yet
    
    if (selectedImageType == "small") {
        if (selectedDevice == "Device 1") {
            chart2_data.values = device1_data.small_image;
            chart2_data.title = "ARTHUR - Small Image Average Performance (ms)";
        } else {
            chart2_data.values = device2_data.small_image;
            chart2_data.title = "KRATOS - Small Image Average Performance (ms)";
        }
    } else if (selectedImageType == "large") {
        if (selectedDevice == "Device 1") {
            chart2_data.values = device1_data.large_image;
            chart2_data.title = "ARTHUR - Large Image Average Performance (ms)";
        } else {
            chart2_data.values = device2_data.large_image;
            chart2_data.title = "KRATOS - Large Image Average Performance (ms)";
        }
    }
    
    update_chart2();
}

void on_device1_clicked(GtkWidget *widget, gpointer data) {
    selectedDevice = "Device 1";
    std::cout << "Selected: Device 1" << std::endl;
    
    if (isFirstSelection) {
        // First time selection - guide user to select image type
        std::string label_text = "Selected: Device 1 - Now choose image size";
        gtk_label_set_text(GTK_LABEL(label), label_text.c_str());
    } else {
        // Subsequent selections - update static chart automatically
        update_static_chart();
        std::string label_text = "Device switched to Device 1 - " + selectedImageType + " image. Run algorithms to compare!";
        gtk_label_set_text(GTK_LABEL(label), label_text.c_str());
        std::cout << "Static chart updated to Device 1 - " << selectedImageType << " image" << std::endl;
    }
}

void on_device2_clicked(GtkWidget *widget, gpointer data) {
    selectedDevice = "Device 2";
    std::cout << "Selected: Device 2" << std::endl;
    
    if (isFirstSelection) {
        // First time selection - guide user to select image type
        std::string label_text = "Selected: Device 2 - Now choose image size";
        gtk_label_set_text(GTK_LABEL(label), label_text.c_str());
    } else {
        // Subsequent selections - update static chart automatically
        update_static_chart();
        std::string label_text = "Device switched to Device 2 - " + selectedImageType + " image. Run algorithms to compare!";
        gtk_label_set_text(GTK_LABEL(label), label_text.c_str());
        std::cout << "Static chart updated to Device 2 - " << selectedImageType << " image" << std::endl;
    }
}

// Callback functions for chart2 buttons
void on_dataset1_clicked(GtkWidget *widget, gpointer data) {
    selectedImageType = "small";
    inputJpg = "images/image.jpg";
    
    // Clear live results only on first selection or when switching image types
    e_time_1 = 0;
    e_time_2 = 0;
    e_time_3 = 0;
    e_time_4 = 0;
    e_time_5 = 0;
    
    // Update static chart based on current device
    update_static_chart();
    
    // Mark that first selection is complete
    isFirstSelection = false;
    
    std::cout << "Image type set to small for " << selectedDevice << std::endl;
    
    // Update label
    std::string label_text = selectedDevice + " - Small Image loaded. Run algorithms to compare!";
    gtk_label_set_text(GTK_LABEL(label), label_text.c_str());
    
    update_chart1();
}

void on_dataset2_clicked(GtkWidget *widget, gpointer data) {
    selectedImageType = "large";
    inputJpg = "images/large_image.jpg";
    
    // Clear live results only on first selection or when switching image types
    e_time_1 = 0;
    e_time_2 = 0;
    e_time_3 = 0;
    e_time_4 = 0;
    e_time_5 = 0;
    
    // Update static chart based on current device
    update_static_chart();
    
    // Mark that first selection is complete
    isFirstSelection = false;
    
    std::cout << "Image type set to large for " << selectedDevice << std::endl;
    
    // Update label
    std::string label_text = selectedDevice + " - Large Image loaded. Run algorithms to compare!";
    gtk_label_set_text(GTK_LABEL(label), label_text.c_str());
    
    update_chart1();
}

// Callback function for button clicks
void on_button1_clicked(GtkWidget *widget, gpointer data) {
    e_time_1 = run_g(inputJpg);
    std::cout << "Sequential: " << e_time_1 << std::endl;
    
    update_chart1();
}

void on_button2_clicked(GtkWidget *widget, gpointer data) {
    e_time_2 = run_og(inputJpg);
    std::cout << "OpenMP: " << e_time_2 << std::endl;
    
    update_chart1();
}

void on_button3_clicked(GtkWidget *widget, gpointer data) {
    e_time_3 = run_ag(inputJpg);
    std::cout << "Advanced CPU: " << e_time_3 << std::endl;
    
    update_chart1();
}

void on_button4_clicked(GtkWidget *widget, gpointer data) {
    e_time_4 = run_gg(inputJpg);
    std::cout << "CUDA - Global Memory: " << e_time_4 << std::endl;
    
    update_chart1();
}

void on_button5_clicked(GtkWidget *widget, gpointer data) {
    e_time_5 = run_sg(inputJpg);
    std::cout << "CUDA - Shared Memory: " << e_time_5 << std::endl;
    
    update_chart1();
}

// Callback for window close
void on_window_closed(GtkWidget *widget, gpointer data) {
    gtk_main_quit();
}

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);
    
    // Create main window
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Gaussian Filtering Performance Comparison");
    gtk_window_set_default_size(GTK_WINDOW(window), 800, 750); // Increased height slightly
    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
    
    // Create main vertical box container
    GtkWidget *main_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_container_add(GTK_CONTAINER(window), main_vbox);
    gtk_container_set_border_width(GTK_CONTAINER(main_vbox), 10);
    
    // Create label
    label = gtk_label_new("Step 1: Select a device to compare against");
    gtk_box_pack_start(GTK_BOX(main_vbox), label, FALSE, FALSE, 0);
    
    // Create horizontal box for device selection buttons
    GtkWidget *device_button_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_pack_start(GTK_BOX(main_vbox), device_button_hbox, FALSE, FALSE, 0);
    
    // Create device selection buttons
    GtkWidget *device1_button = gtk_button_new_with_label("Device 1: ARTHUR");
    GtkWidget *device2_button = gtk_button_new_with_label("Device 2: KRATOS");
    
    // Style the device buttons
    gtk_widget_set_size_request(device1_button, 200, 35);
    gtk_widget_set_size_request(device2_button, 200, 35);
    
    gtk_box_pack_start(GTK_BOX(device_button_hbox), device1_button, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(device_button_hbox), device2_button, FALSE, FALSE, 0);
    
    // Add separator
    GtkWidget *separator1 = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_pack_start(GTK_BOX(main_vbox), separator1, FALSE, FALSE, 5);
    
    // Create horizontal box for chart2 control buttons
    GtkWidget *chart2_button_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_pack_start(GTK_BOX(main_vbox), chart2_button_hbox, FALSE, FALSE, 0);
    
    // Create chart2 control buttons
    GtkWidget *dataset1_button = gtk_button_new_with_label("Small Image");
    GtkWidget *dataset2_button = gtk_button_new_with_label("Large Image");
    
    // Style the chart2 buttons
    gtk_widget_set_size_request(dataset1_button, 150, 30);
    gtk_widget_set_size_request(dataset2_button, 150, 30);
    
    gtk_box_pack_start(GTK_BOX(chart2_button_hbox), dataset1_button, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(chart2_button_hbox), dataset2_button, FALSE, FALSE, 0);
    
    // Add separator
    GtkWidget *separator2 = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_pack_start(GTK_BOX(main_vbox), separator2, FALSE, FALSE, 5);
    
    // Create horizontal box for main execution buttons
    GtkWidget *button_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_pack_start(GTK_BOX(main_vbox), button_hbox, FALSE, FALSE, 0);
    
    // Create buttons
    GtkWidget *button1 = gtk_button_new_with_label("Sequential");
    GtkWidget *button2 = gtk_button_new_with_label("OpenMP");
    GtkWidget *button3 = gtk_button_new_with_label("Advanced CPU");
    GtkWidget *button4 = gtk_button_new_with_label("CUDA - Global Memory");
    GtkWidget *button5 = gtk_button_new_with_label("CUDA - Shared Memory");
    
    gtk_box_pack_start(GTK_BOX(button_hbox), button1, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(button_hbox), button2, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(button_hbox), button3, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(button_hbox), button4, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(button_hbox), button5, TRUE, TRUE, 0);
    
    // Create horizontal box for charts
    GtkWidget *charts_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_box_pack_start(GTK_BOX(main_vbox), charts_hbox, TRUE, TRUE, 0);
    
    // Create drawing areas for charts
    chart1_area = gtk_drawing_area_new();
    chart2_area = gtk_drawing_area_new();
    
    gtk_widget_set_size_request(chart1_area, 380, 300);
    gtk_widget_set_size_request(chart2_area, 380, 300);
    
    // Add frames around charts for better visual separation
    GtkWidget *frame1 = gtk_frame_new("Dynamic Results");
    GtkWidget *frame2 = gtk_frame_new("Static Results");
    
    gtk_container_add(GTK_CONTAINER(frame1), chart1_area);
    gtk_container_add(GTK_CONTAINER(frame2), chart2_area);
    
    gtk_box_pack_start(GTK_BOX(charts_hbox), frame1, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(charts_hbox), frame2, TRUE, TRUE, 0);
    
    // Connect drawing signals
    g_signal_connect(G_OBJECT(chart1_area), "draw", G_CALLBACK(draw_chart), &chart1_data);
    g_signal_connect(G_OBJECT(chart2_area), "draw", G_CALLBACK(draw_chart), &chart2_data);
    
    // Connect device selection button signals
    g_signal_connect(device1_button, "clicked", G_CALLBACK(on_device1_clicked), NULL);
    g_signal_connect(device2_button, "clicked", G_CALLBACK(on_device2_clicked), NULL);
    
    // Connect chart2 button signals
    g_signal_connect(dataset1_button, "clicked", G_CALLBACK(on_dataset1_clicked), NULL);
    g_signal_connect(dataset2_button, "clicked", G_CALLBACK(on_dataset2_clicked), NULL);
    
    // Connect main execution button signals
    g_signal_connect(button1, "clicked", G_CALLBACK(on_button1_clicked), NULL);
    g_signal_connect(button2, "clicked", G_CALLBACK(on_button2_clicked), NULL);
    g_signal_connect(button3, "clicked", G_CALLBACK(on_button3_clicked), NULL);
    g_signal_connect(button4, "clicked", G_CALLBACK(on_button4_clicked), NULL);
    g_signal_connect(button5, "clicked", G_CALLBACK(on_button5_clicked), NULL);
    g_signal_connect(window, "destroy", G_CALLBACK(on_window_closed), NULL);
    
    // Show all widgets
    gtk_widget_show_all(window);
    
    // Start main loop
    gtk_main();
    
    return 0;
}
