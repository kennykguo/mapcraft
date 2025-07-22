# compilers
CXX = g++
NVCC = nvcc
MKDIR = mkdir -p

# configuration
CONF ?= release
BUILD_DIR = build/$(CONF)

# source files - main.cpp includes 3d_rendering.cpp which includes all other files
MAIN_SRCS = main.cpp
CUDA_SRCS = cuda_frustum_culling.cu cuda_spatial_grid.cu
C_SRCS = glad/src/glad.c

# object files
OBJS = \
    $(addprefix $(BUILD_DIR)/, $(MAIN_SRCS:.cpp=.o)) \
    $(addprefix $(BUILD_DIR)/, $(CUDA_SRCS:.cu=.o)) \
    $(addprefix $(BUILD_DIR)/, $(C_SRCS:.c=.o))

# compiler flags
CXXFLAGS = \
    -std=c++17 \
    -fopenmp \
    -Iglad/include \
    -Iglm \
    -Iglfw/include \
    -I. \
    -I/opt/cuda/include

NVCCFLAGS = \
    -std=c++17 \
    -Xcompiler="-fopenmp" \
    -Iglad/include \
    -Iglm \
    -Iglfw/include \
    -I. \
    --expt-relaxed-constexpr \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80

LDFLAGS = \
    -L/opt/cuda/lib64 \
    -lcudart \
    -lGL \
    -ldl \
    -lpthread \
    -lm \
    -lz \
    -Lglfw/src \
    -lglfw

# target
TARGET = 3d

# build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

# compilation rules
$(BUILD_DIR)/%.o: %.cpp
	@$(MKDIR) $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@$(MKDIR) $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@$(MKDIR) $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean