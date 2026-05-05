cmake_minimum_required(VERSION 3.18)

# ── Strip conda's compiler_compat linker shim (BEFORE project()) ──────────────
# project() triggers compiler detection which reads CFLAGS/CXXFLAGS/LDFLAGS from
# the environment — so stripping must happen here, before that call.
# When the conda-forge 'gcc' metapackage is installed, conda activation injects:
#   CFLAGS/CXXFLAGS/LDFLAGS: -B <env>/compiler_compat  (redirects ld to conda sysroot)
#   CPATH/CPLUS_INCLUDE_PATH: <env>/lib/gcc/.../include/c++ (GCC 15 headers)
# Both cause build failures on systems with older glibc (e.g. openSUSE Leap 15.x).
# This block is a no-op when building outside conda or without the gcc metapackage.
foreach(_ev CFLAGS CXXFLAGS LDFLAGS)
    if(DEFINED ENV{${_ev}})
        string(REGEX REPLACE "-B ?[^ ]*compiler_compat[^ ]*" "" _cleaned "$ENV{${_ev}}")
        string(STRIP "${_cleaned}" _cleaned)
        set(ENV{${_ev}} "${_cleaned}")
    endif()
endforeach()
foreach(_ev CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH)
    if(DEFINED ENV{${_ev}})
        string(REPLACE ":" ";" _parts "$ENV{${_ev}}")
        set(_keep "")
        foreach(_p IN LISTS _parts)
            if(NOT _p MATCHES "conda-linux-gnu|compiler_compat")
                list(APPEND _keep "${_p}")
            endif()
        endforeach()
        string(JOIN ":" _cleaned "${_keep}")
        set(ENV{${_ev}} "${_cleaned}")
    endif()
endforeach()
unset(_cleaned)
unset(_keep)
unset(_p)
unset(_parts)
unset(_ev)

project(pygrog_torch CXX)
# ── Python ────────────────────────────────────────────────────────────────────
# scikit-build-core provides Python_EXECUTABLE; honour it.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# ── PyTorch / libtorch ────────────────────────────────────────────────────────
# Ask the target Python where its Torch CMake packages live.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_query_result
)
if(NOT _torch_query_result EQUAL 0 OR TORCH_CMAKE_PREFIX STREQUAL "")
    message(FATAL_ERROR
        "PyTorch not found in the build Python (${Python_EXECUTABLE}).\n"
        "This usually means pip's build isolation stripped your environment.\n"
        "Fix: run pip with --no-build-isolation so the active venv/conda env "
        "is visible during the CMake configure step:\n"
        "  pip install --no-build-isolation -e .\n"
        "  pip install --no-build-isolation pygrog\n"
        "Alternatively, install torch inside the build environment first:\n"
        "  pip install torch && pip install pygrog"
    )
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

# ── Auto-locate nvcc before find_package(Torch) ───────────────────────────────
# A CUDA-built torch ships Caffe2Config.cmake which calls enable_language(CUDA)
# unconditionally.  That call fails if CMAKE_CUDA_COMPILER is unset AND nvcc is
# not in PATH (common when CUDA is installed under /usr/local/cuda but that
# directory was not added to PATH).  We probe standard installation prefixes so
# the build works without requiring the user to manually export PATH first.
if(NOT CMAKE_CUDA_COMPILER)
    find_program(_auto_nvcc
        NAMES nvcc
        PATHS
            /usr/local/cuda/bin
            /usr/local/cuda-13.0/bin
            /usr/local/cuda-12.8/bin
            /usr/local/cuda-12.6/bin
            /usr/local/cuda-12/bin
            /opt/cuda/bin
        NO_DEFAULT_PATH
    )
    if(_auto_nvcc)
        set(CMAKE_CUDA_COMPILER "${_auto_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "Auto-located nvcc: ${CMAKE_CUDA_COMPILER} "
                       "(add $(dirname ${CMAKE_CUDA_COMPILER}) to PATH to suppress this message)")
    endif()
    unset(_auto_nvcc CACHE)
endif()

# ── Strip conda Python sysconfig flags from CMAKE_CXX_FLAGS ──────────────────
# conda-forge bakes "-pthread -B <env>/compiler_compat ..." into the Python
# interpreter's sysconfig CFLAGS.  FindPython(Development.Module) propagates
# those into CMAKE_CXX_FLAGS.  Caffe2Config.cmake then inherits them for its
# detect_cuda_version test compilation, where the conda-forge ld shim
# (__vdso_clock_gettime@GLIBC_PRIVATE) fails on older host glibc.
# Strip the offending flag here, after FindPython but before find_package(Torch).
foreach(_v CMAKE_C_FLAGS CMAKE_CXX_FLAGS
           CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
    if(DEFINED ${_v})
        string(REGEX REPLACE "-B ?[^ ]*compiler_compat[^ ]*" "" ${_v} "${${_v}}")
        string(STRIP "${${_v}}" ${_v})
    endif()
endforeach()
unset(_v)

find_package(Torch REQUIRED)

# ── CUDA (optional) ───────────────────────────────────────────────────────────
# Disable CUDA via cmake define (-DPYGROG_NO_CUDA=ON) or env var (PYGROG_NO_CUDA=1).
if(PYGROG_NO_CUDA OR DEFINED ENV{PYGROG_NO_CUDA})
    set(_skip_cuda ON)
endif()

if(NOT _skip_cuda)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(USE_CUDA ON)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endif()

if((PYGROG_FORCE_CUDA OR DEFINED ENV{PYGROG_FORCE_CUDA}) AND NOT USE_CUDA)
    message(FATAL_ERROR
        "PYGROG_FORCE_CUDA is set but no CUDA compiler was found.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    csrc/torch/module.cpp
    csrc/torch/grog_interp.cpp
    csrc/torch/sparse_ops.cpp
    csrc/torch/sparse_ops_avx2.cpp
    csrc/torch/sparse_ops_avx512.cpp
    csrc/torch/toep_psf.cpp
)

if(USE_CUDA)
    list(APPEND SOURCES
        csrc/torch/grog_interp_cuda.cu
        csrc/torch/sparse_ops_cuda.cu
        csrc/torch/toep_psf_cuda.cu
    )
endif()

# ── Extension target ──────────────────────────────────────────────────────────
python_add_library(_pygrog_torch MODULE WITH_SOABI ${SOURCES})
# torch_python provides pybind11 type_casters for at::Tensor (libtorch_python).
# It is NOT included in TORCH_LIBRARIES; use find_library so the linker can
# find the platform-appropriate file (.so on Linux, .dylib on macOS, .lib on Windows)
# regardless of whether torch/lib is in the default search path.
cmake_path(GET TORCH_CMAKE_PREFIX PARENT_PATH _tp_share)
cmake_path(GET _tp_share PARENT_PATH _tp_root)
find_library(_torch_python_lib
    NAMES torch_python
    HINTS "${_tp_root}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)
unset(_tp_share)
unset(_tp_root)
target_link_libraries(_pygrog_torch PRIVATE ${TORCH_LIBRARIES} ${_torch_python_lib})
unset(_torch_python_lib CACHE)
target_include_directories(_pygrog_torch PRIVATE ${TORCH_INCLUDE_DIRS})
# PyTorch's cpp_extension.py injects -DTORCH_EXTENSION_NAME=<name> automatically.
# With plain CMake we must define it ourselves so PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
# expands to PyInit__pygrog_torch (without this it exports PyInit_TORCH_EXTENSION_NAME).
target_compile_definitions(_pygrog_torch PRIVATE TORCH_EXTENSION_NAME=_pygrog_torch)

# ── Compile flags ─────────────────────────────────────────────────────────────
# GCC/Clang: the AVX files enable their ISA via #pragma GCC target — no
# per-file flag needed.  MSVC requires explicit /arch: flags.
if(MSVC)
    target_compile_options(_pygrog_torch PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /openmp:experimental>
    )
    set_source_files_properties(csrc/torch/sparse_ops_avx2.cpp PROPERTIES
        COMPILE_FLAGS "/arch:AVX2"
    )
    set_source_files_properties(csrc/torch/sparse_ops_avx512.cpp PROPERTIES
        COMPILE_FLAGS "/arch:AVX512"
    )
else()
    target_compile_options(_pygrog_torch PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fopenmp>
    )
    target_link_options(_pygrog_torch PRIVATE -fopenmp)
    # Explicitly set the ISA for the AVX dispatch TUs so GCC 13+'s stricter
    # always_inline target-mismatch check doesn't fire when std::allocator
    # (always_inline) is instantiated under a #pragma GCC target("avx512f").
    set_source_files_properties(csrc/torch/sparse_ops_avx2.cpp PROPERTIES
        COMPILE_FLAGS "-mavx2 -mfma"
    )
    set_source_files_properties(csrc/torch/sparse_ops_avx512.cpp PROPERTIES
        COMPILE_FLAGS "-mavx512f"
    )
endif()

# ── CUDA settings ─────────────────────────────────────────────────────────────
if(USE_CUDA)
    target_compile_definitions(_pygrog_torch PRIVATE COMPILE_WITH_CUDA=1)
    target_compile_options(_pygrog_torch PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --expt-relaxed-constexpr
        >
    )
    set_property(TARGET _pygrog_torch PROPERTY
        CUDA_ARCHITECTURES 70 80 86 89 90
    )
endif()

# ── RPATH ─────────────────────────────────────────────────────────────────────
# Embed the path to torch's shared libraries so the extension can be loaded
# without LD_LIBRARY_PATH.  The $ORIGIN token is resolved at load time relative
# to the .so's location; we walk up to site-packages then into torch/lib/.
# Also embed the CUDA lib64 path when building with CUDA.
#
# TORCH_CMAKE_PREFIX = <site-packages>/torch/share/cmake
# Go up twice: .../torch/share/cmake -> .../torch/share -> .../torch
cmake_path(GET TORCH_CMAKE_PREFIX PARENT_PATH _torch_share)  # .../torch/share
cmake_path(GET _torch_share PARENT_PATH _torch_root)          # .../torch
set(_torch_lib "${_torch_root}/lib")

set(_rpath_entries "$ORIGIN/../torch/lib" "${_torch_lib}")
if(USE_CUDA)
    list(APPEND _rpath_entries "/usr/local/cuda/lib64")
endif()

# Must use set_target_properties (not CMAKE_INSTALL_RPATH) because the target
# was already created; CMAKE_INSTALL_RPATH only sets the default at creation time.
# BUILD_WITH_INSTALL_RPATH TRUE ensures the build-tree .so also carries the RPATH,
# which is required for editable installs where the .so is used in-place.
set_target_properties(_pygrog_torch PROPERTIES
    INSTALL_RPATH "${_rpath_entries}"
    BUILD_WITH_INSTALL_RPATH TRUE
)
unset(_rpath_entries)
unset(_torch_share)
unset(_torch_root)
unset(_torch_lib)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS _pygrog_torch DESTINATION pygrog)
