CXX = clang++
CXXFLAGS = -std=c++11 -Wall -Wextra

# Directories
SRC_DIR = .
BUILD_DIR = build

# Target executable
TARGET = $(BUILD_DIR)/kmeans_serial

# Source files
SRCS = main.cpp kmeans.cpp data_generator.cpp

# Object files (with build directory path)
OBJS = $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

# Header files
HEADERS = kmeans.h data_generator.h

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	mkdir -p $(BUILD_DIR)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compiling
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Run
run: $(TARGET)
	$(TARGET)

.PHONY: all clean run directories