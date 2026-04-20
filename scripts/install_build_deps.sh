#!/usr/bin/env bash
# ============================================================================
# install_build_deps.sh — Install C++ (and optionally CUDA) build toolchain.
#
# Supports: Ubuntu/Debian, Fedora/RHEL/Rocky, Arch, macOS (Homebrew).
# CUDA toolkit is optional — pass --cuda to install it.
#
# Usage:
#   ./scripts/install_build_deps.sh          # C++ toolchain only
#   ./scripts/install_build_deps.sh --cuda   # C++ + CUDA toolkit
#
# On Linux this script needs root privileges (sudo).
# On macOS it uses Homebrew (no sudo required for brew itself).
# ============================================================================
set -euo pipefail

INSTALL_CUDA=0
CUDA_VERSION="12-6"  # default CUDA toolkit version

for arg in "$@"; do
    case "$arg" in
        --cuda)         INSTALL_CUDA=1 ;;
        --cuda-version=*) CUDA_VERSION="${arg#*=}" ;;
        -h|--help)
            echo "Usage: $0 [--cuda] [--cuda-version=12-6]"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ---- helpers ---------------------------------------------------------------
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
fail()  { printf '\033[1;31m[FAIL]\033[0m  %s\n' "$*"; exit 1; }

has_cmd() { command -v "$1" &>/dev/null; }

need_sudo() {
    if [[ $EUID -ne 0 ]]; then
        if has_cmd sudo; then
            SUDO=sudo
        else
            fail "This script requires root privileges. Run as root or install sudo."
        fi
    else
        SUDO=""
    fi
}

# ---- detect OS & package manager -------------------------------------------
detect_os() {
    if [[ "$OSTYPE" == darwin* ]]; then
        OS=macos
    elif [[ -f /etc/os-release ]]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        case "$ID" in
            ubuntu|debian|pop|linuxmint) OS=debian ;;
            fedora|rhel|rocky|centos|almalinux) OS=fedora ;;
            arch|manjaro|endeavouros) OS=arch ;;
            *) OS=unknown ;;
        esac
    else
        OS=unknown
    fi
}

# ---- check what's already installed ----------------------------------------
check_existing() {
    info "Checking existing tools..."

    if has_cmd g++; then
        GXX_VER=$(g++ -dumpversion 2>/dev/null || echo "?")
        ok "g++ found (version $GXX_VER)"
        HAVE_GXX=1
    elif has_cmd c++ && c++ --version 2>&1 | grep -qi clang; then
        CLANG_VER=$(c++ -dumpversion 2>/dev/null || echo "?")
        ok "clang++ found (version $CLANG_VER)"
        HAVE_GXX=1  # clang works fine
    else
        warn "No C++ compiler found"
        HAVE_GXX=0
    fi

    if has_cmd ninja; then
        ok "ninja found"
        HAVE_NINJA=1
    else
        HAVE_NINJA=0
    fi

    if has_cmd nvcc; then
        NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "?")
        ok "nvcc found (CUDA $NVCC_VER)"
        HAVE_NVCC=1
    else
        HAVE_NVCC=0
    fi
}

# ---- Debian / Ubuntu -------------------------------------------------------
install_debian() {
    need_sudo
    local pkgs=()

    if [[ $HAVE_GXX -eq 0 ]]; then
        pkgs+=(build-essential g++)
    fi
    pkgs+=(ninja-build)

    if [[ ${#pkgs[@]} -gt 0 ]]; then
        info "Installing: ${pkgs[*]}"
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq "${pkgs[@]}"
    else
        ok "C++ toolchain already present"
    fi

    if [[ $INSTALL_CUDA -eq 1 && $HAVE_NVCC -eq 0 ]]; then
        info "Installing CUDA toolkit ${CUDA_VERSION} ..."
        # NVIDIA apt repo (works for Ubuntu 20.04+)
        if ! has_cmd nvidia-smi && ! dpkg -l | grep -q cuda-keyring; then
            local DISTRO
            DISTRO=$(. /etc/os-release && echo "${ID}${VERSION_ID}" | tr -d '.')
            local ARCH
            ARCH=$(dpkg --print-architecture)
            local KEYRING="cuda-keyring_1.1-1_all.deb"
            local URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/${KEYRING}"
            info "Adding NVIDIA apt repository..."
            $SUDO apt-get install -y -qq wget
            wget -q "$URL" -O "/tmp/$KEYRING"
            $SUDO dpkg -i "/tmp/$KEYRING"
            $SUDO apt-get update -qq
        fi
        $SUDO apt-get install -y -qq "cuda-toolkit-${CUDA_VERSION}"
        ok "CUDA toolkit installed. You may need to add /usr/local/cuda/bin to PATH."
    elif [[ $INSTALL_CUDA -eq 1 ]]; then
        ok "CUDA (nvcc) already present"
    fi
}

# ---- Fedora / RHEL ---------------------------------------------------------
install_fedora() {
    need_sudo
    local pkgs=()

    if [[ $HAVE_GXX -eq 0 ]]; then
        pkgs+=(gcc-c++ make)
    fi
    pkgs+=(ninja-build)

    if [[ ${#pkgs[@]} -gt 0 ]]; then
        info "Installing: ${pkgs[*]}"
        $SUDO dnf install -y -q "${pkgs[@]}"
    else
        ok "C++ toolchain already present"
    fi

    if [[ $INSTALL_CUDA -eq 1 && $HAVE_NVCC -eq 0 ]]; then
        info "Installing CUDA toolkit (via dnf) ..."
        $SUDO dnf config-manager --add-repo \
            "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo" 2>/dev/null || true
        $SUDO dnf install -y -q "cuda-toolkit-${CUDA_VERSION}"
        ok "CUDA toolkit installed."
    elif [[ $INSTALL_CUDA -eq 1 ]]; then
        ok "CUDA (nvcc) already present"
    fi
}

# ---- Arch Linux ------------------------------------------------------------
install_arch() {
    need_sudo

    if [[ $HAVE_GXX -eq 0 ]]; then
        info "Installing base-devel..."
        $SUDO pacman -S --noconfirm --needed base-devel
    else
        ok "C++ toolchain already present"
    fi
    $SUDO pacman -S --noconfirm --needed ninja

    if [[ $INSTALL_CUDA -eq 1 && $HAVE_NVCC -eq 0 ]]; then
        info "Installing cuda..."
        $SUDO pacman -S --noconfirm --needed cuda
    elif [[ $INSTALL_CUDA -eq 1 ]]; then
        ok "CUDA (nvcc) already present"
    fi
}

# ---- macOS (Homebrew) ------------------------------------------------------
install_macos() {
    if ! has_cmd brew; then
        fail "Homebrew not found. Install it from https://brew.sh"
    fi

    # Xcode Command Line Tools provide clang++
    if [[ $HAVE_GXX -eq 0 ]]; then
        info "Installing Xcode Command Line Tools..."
        xcode-select --install 2>/dev/null || true
    else
        ok "C++ compiler already present"
    fi

    if [[ $HAVE_NINJA -eq 0 ]]; then
        info "Installing ninja..."
        brew install ninja
    fi

    if [[ $INSTALL_CUDA -eq 1 ]]; then
        warn "CUDA is not supported on macOS (Apple Silicon). Skipping."
    fi
}

# ---- main ------------------------------------------------------------------
detect_os
check_existing

case "$OS" in
    debian) install_debian ;;
    fedora) install_fedora ;;
    arch)   install_arch ;;
    macos)  install_macos ;;
    *)      fail "Unsupported OS. Please install a C++17 compiler manually." ;;
esac

echo ""
ok "Build dependencies installed successfully."
if [[ $INSTALL_CUDA -eq 1 && $HAVE_NVCC -eq 0 ]]; then
    info "Remember to add CUDA to your PATH:"
    info "  export PATH=/usr/local/cuda/bin:\$PATH"
fi
info "You can now build pygrog with:  pip install -e ."
