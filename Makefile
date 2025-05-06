# compilers
CXX = g++
NVCC = nvcc
MKDIR = mkdir -p

# configuration
CONF ?= release
BUILD_DIR = build/$(CONF)

# source files - need to compile main.cpp aswell as 3d_rendering.cpp, rest will follow
MAIN_SRCS = main.cpp 3d_rendering.cpp
CUDA_SRCS = cuda_frustum_culling.cu cuda_spatial_grid.cu
C_SRCS = glad/src/glad.c

# exclude files that are included in 3d_rendering.cpp
EXCLUDED_SRCS = $(filter-out $(MAIN_SRCS), $(wildcard *.cpp))

# file discovery with exclusions
CPP_SRCS = $(filter-out $(EXCLUDED_SRCS), $(wildcard *.cpp))
C_SRCS = glad/src/glad.c

# object files
OBJS = \
    $(addprefix $(BUILD_DIR)/, $(CPP_SRCS:.cpp=.o)) \
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