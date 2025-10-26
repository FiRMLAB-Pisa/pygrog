# CMake toolchain for PyGROG with optimized settings
# This file contains platform-specific optimization settings

# Set optimization flags based on build type and compiler
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        # MSVC optimizations
        set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /Oi /Ot /Oy /GL /DNDEBUG")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/LTCG")
        
        # SIMD flags for MSVC
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " /arch:AVX2")
        
    else()
        # GCC/Clang optimizations
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
        
        # Architecture-specific optimizations
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            # x86_64 optimizations
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native -mtune=native")
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -mavx2 -mfma -msse4.2")
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -fopenmp-simd")
            
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            # ARM64 optimizations
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -mcpu=native")
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -ftree-vectorize")
            
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
            # ARM32 optimizations
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -mfpu=neon")
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -ftree-vectorize")
        endif()
        
        # Additional performance flags
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -ffast-math")
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -funroll-loops")
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -flto")
        
        # Linker optimizations
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto -O3")
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "-flto -O3")
    endif()
    
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(MSVC)
        set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /RTC1 /MDd")
    else()
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")
        
        # Optional: Enable sanitizers in debug mode
        if(ENABLE_SANITIZERS)
            string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=address")
            string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fsanitize=undefined")
            set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address -fsanitize=undefined")
            set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "-fsanitize=address -fsanitize=undefined")
        endif()
    endif()
endif()

# Compiler-specific warnings
if(MSVC)
    # MSVC warnings
    add_compile_options(/W4)
    # Disable specific warnings
    add_compile_options(/wd4244 /wd4267 /wd4996)
else()
    # GCC/Clang warnings
    add_compile_options(-Wall -Wextra -Wpedantic)
    # Disable specific warnings that are too strict for pybind11
    add_compile_options(-Wno-unused-parameter -Wno-sign-compare)
endif()

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    add_definitions(-DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
    
elseif(APPLE)
    # macOS-specific settings
    set(CMAKE_MACOSX_RPATH ON)
    
elseif(UNIX)
    # Linux-specific settings
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Function to display optimization summary
function(display_optimization_summary)
    message(STATUS "")
    message(STATUS "Optimization Summary:")
    message(STATUS "====================")
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS_RELEASE}")
        message(STATUS "Linker flags: ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS_DEBUG}")
        message(STATUS "Linker flags: ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
    endif()
    message(STATUS "")
endfunction()

# Cache variables for user configuration
set(ENABLE_NATIVE_ARCH ON CACHE BOOL "Enable native architecture optimizations")
set(ENABLE_FAST_MATH ON CACHE BOOL "Enable fast math optimizations") 
set(ENABLE_LTO ON CACHE BOOL "Enable Link Time Optimization")
set(ENABLE_SANITIZERS OFF CACHE BOOL "Enable sanitizers in debug builds")

# Apply user configuration
if(NOT ENABLE_NATIVE_ARCH AND NOT MSVC)
    string(REPLACE "-march=native" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-mtune=native" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-mcpu=native" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

if(NOT ENABLE_FAST_MATH AND NOT MSVC)
    string(REPLACE "-ffast-math" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

if(NOT ENABLE_LTO AND NOT MSVC)
    string(REPLACE "-flto" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-flto" "" CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
endif()