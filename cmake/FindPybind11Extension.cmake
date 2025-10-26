# CMake configuration for finding pybind11 and setting up Python extension

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Try to find pybind11 via pip install location
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_FIND_RESULT
    )
    
    if(pybind11_FIND_RESULT EQUAL 0)
        set(pybind11_DIR "${pybind11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Please install it with: pip install pybind11")
    endif()
endif()

# Function to add Python extension with proper settings
function(add_python_extension target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDE_DIRS;COMPILE_DEFINITIONS" ${ARGN})
    
    # Create the pybind11 module
    pybind11_add_module(${target_name} ${ARG_SOURCES})
    
    # Set properties
    set_target_properties(${target_name} PROPERTIES
        PREFIX ""
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    
    # Add include directories
    if(ARG_INCLUDE_DIRS)
        target_include_directories(${target_name} PRIVATE ${ARG_INCLUDE_DIRS})
    endif()
    
    # Add compile definitions
    if(ARG_COMPILE_DEFINITIONS)
        target_compile_definitions(${target_name} PRIVATE ${ARG_COMPILE_DEFINITIONS})
    endif()
    
    # Platform-specific optimizations
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /O2 /arch:AVX2)
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${target_name} PRIVATE /Oi /Ot /Oy /GL)
            target_link_options(${target_name} PRIVATE /LTCG)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE -O3 -Wall -Wextra)
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${target_name} PRIVATE 
                -march=native -mtune=native -mavx2 -mfma -fopenmp-simd
            )
        endif()
    endif()
endfunction()

# Macro to detect SIMD capabilities
macro(detect_simd_support)
    include(CheckCXXCompilerFlag)
    
    # Initialize SIMD flags
    set(SIMD_FLAGS "")
    set(SIMD_DEFINITIONS "")
    
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|X86")
        # Check AVX-512
        check_cxx_compiler_flag("-mavx512f" HAS_AVX512F)
        if(HAS_AVX512F)
            list(APPEND SIMD_FLAGS "-mavx512f")
            list(APPEND SIMD_DEFINITIONS "HAS_AVX512F=1")
            message(STATUS "AVX-512 support: YES")
        else()
            message(STATUS "AVX-512 support: NO")
        endif()
        
        # Check AVX2
        check_cxx_compiler_flag("-mavx2" HAS_AVX2)
        if(HAS_AVX2)
            list(APPEND SIMD_FLAGS "-mavx2")
            list(APPEND SIMD_DEFINITIONS "HAS_AVX2=1")
            message(STATUS "AVX2 support: YES")
        else()
            message(STATUS "AVX2 support: NO")
        endif()
        
        # Check SSE4.2
        check_cxx_compiler_flag("-msse4.2" HAS_SSE42)
        if(HAS_SSE42)
            list(APPEND SIMD_FLAGS "-msse4.2")
            list(APPEND SIMD_DEFINITIONS "HAS_SSE42=1")
            message(STATUS "SSE4.2 support: YES")
        else()
            message(STATUS "SSE4.2 support: NO")
        endif()
        
        # Always enable SSE2 for x86_64
        list(APPEND SIMD_FLAGS "-msse2")
        list(APPEND SIMD_DEFINITIONS "HAS_SSE2=1")
        
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM")
        # ARM NEON support
        check_cxx_compiler_flag("-mfpu=neon" HAS_NEON)
        if(HAS_NEON)
            list(APPEND SIMD_FLAGS "-mfpu=neon")
            list(APPEND SIMD_DEFINITIONS "HAS_NEON=1")
            message(STATUS "ARM NEON support: YES")
        else()
            message(STATUS "ARM NEON support: NO")
        endif()
    endif()
    
    # Make variables available globally using CACHE
    set(SIMD_COMPILE_FLAGS ${SIMD_FLAGS} CACHE INTERNAL "SIMD compilation flags")
    set(SIMD_COMPILE_DEFINITIONS ${SIMD_DEFINITIONS} CACHE INTERNAL "SIMD compile definitions")
endmacro()