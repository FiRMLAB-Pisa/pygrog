#!/usr/bin/env bash
# ============================================================================
# install_build_deps.sh — Install C++ (and optionally CUDA) build toolchain.
#
# Supports: Ubuntu/Debian, Fedora/RHEL/Rocky, openSUSE/SLES, Arch,
#           macOS (Homebrew).
# CUDA toolkit is optional — pass --cuda to install it.
# Supported CUDA versions (matching PyTorch): 12.6, 12.8, 13.0
#
# Usage:
#   Linux   (requires sudo):
#     sudo ./scripts/install_build_deps.sh
#     sudo ./scripts/install_build_deps.sh --cuda
#     sudo ./scripts/install_build_deps.sh --cuda --cuda-version=12.8
#
#   macOS   (Homebrew, no sudo):
#     ./scripts/install_build_deps.sh
#
#   Windows:
#     Run scripts\install_build_deps.ps1 from an Administrator PowerShell.
# ============================================================================
set -euo pipefail

INSTALL_CUDA=0
CUDA_VERSION="12.6"   # default; must be one of: 12.6, 12.8, 13.0

for arg in "$@"; do
    case "$arg" in
        --cuda)             INSTALL_CUDA=1 ;;
        --cuda-version=*)   CUDA_VERSION="${arg#*=}" ;;
        -h|--help)
            echo "Usage: $0 [--cuda] [--cuda-version=12.6|12.8|13.0]"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# Validate and normalise CUDA version.
# Accepted dot-separated form (e.g. 12.6) → dash-separated for pkg names.
case "$CUDA_VERSION" in
    12.6|12-6) CUDA_VERSION="12.6"; CUDA_PKG_VER="12-6" ;;
    12.8|12-8) CUDA_VERSION="12.8"; CUDA_PKG_VER="12-8" ;;
    13.0|13-0) CUDA_VERSION="13.0"; CUDA_PKG_VER="13-0" ;;
    *)
        echo "Unsupported CUDA version: $CUDA_VERSION"
        echo "Valid choices: 12.6, 12.8, 13.0"
        exit 1
        ;;
esac

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
            opensuse-leap|opensuse-tumbleweed|sles|suse) OS=opensuse ;;
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
        GXX_VER=$(g++ -dumpversion 2>/dev/null || echo "0")
        GXX_MAJOR="${GXX_VER%%.*}"
        if [[ "$GXX_MAJOR" -ge 9 ]] 2>/dev/null; then
            ok "g++ found (version $GXX_VER)"
            HAVE_GXX=1
        else
            warn "g++ found but version $GXX_VER is too old (PyTorch 2.x requires GCC ≥ 9)"
            HAVE_GXX=0
        fi
    elif has_cmd c++ && c++ --version 2>&1 | grep -qi clang; then
        CLANG_VER=$(c++ -dumpversion 2>/dev/null || echo "?")
        ok "clang++ found (version $CLANG_VER)"
        HAVE_GXX=1  # clang works fine
    else
        warn "No C++ compiler found"
        HAVE_GXX=0
    fi

    if has_cmd cmake; then
        CMAKE_VER=$(cmake --version 2>/dev/null | head -1 | grep -oP '[\d.]+' || echo "?")
        ok "cmake found (version $CMAKE_VER)"
        HAVE_CMAKE=1
    else
        HAVE_CMAKE=0
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
    [[ $HAVE_CMAKE -eq 0 ]] && pkgs+=(cmake)
    [[ $HAVE_NINJA -eq 0 ]] && pkgs+=(ninja-build)

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
        if ! dpkg -l 2>/dev/null | grep -q cuda-keyring; then
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
        $SUDO apt-get install -y -qq "cuda-toolkit-${CUDA_PKG_VER}"
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
    [[ $HAVE_CMAKE -eq 0 ]] && pkgs+=(cmake)
    [[ $HAVE_NINJA -eq 0 ]] && pkgs+=(ninja-build)

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
        $SUDO dnf install -y -q "cuda-toolkit-${CUDA_PKG_VER}"
        ok "CUDA toolkit installed."
    elif [[ $INSTALL_CUDA -eq 1 ]]; then
        ok "CUDA (nvcc) already present"
    fi
}

# ---- openSUSE / SLES -------------------------------------------------------
install_opensuse() {
    need_sudo
    local pkgs=()

    if [[ $HAVE_GXX -eq 0 ]]; then
        # openSUSE Leap 15.x ships gcc-c++ = GCC 7, which is too old for PyTorch 2.x.
        # Install the versioned gcc13 packages explicitly.
        pkgs+=(gcc13 gcc13-c++ make)
    fi
    [[ $HAVE_CMAKE -eq 0 ]] && pkgs+=(cmake)
    [[ $HAVE_NINJA -eq 0 ]] && pkgs+=(ninja)

    if [[ ${#pkgs[@]} -gt 0 ]]; then
        info "Installing: ${pkgs[*]}"
        $SUDO zypper --non-interactive install "${pkgs[@]}"
    else
        ok "C++ toolchain already present"
    fi

    # Register gcc-13/g++-13 as the system default via update-alternatives
    # so that subsequent cmake/pip invocations pick it up automatically.
    if [[ $HAVE_GXX -eq 0 ]] && has_cmd gcc-13 && has_cmd g++-13; then
        $SUDO update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 \
            --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
            --slave /usr/bin/cc  cc  /usr/bin/gcc-13 \
            --slave /usr/bin/c++ c++ /usr/bin/g++-13 2>/dev/null || true
        ok "gcc-13/g++-13 registered as default compiler (g++ -dumpversion: $(g++ -dumpversion 2>/dev/null || echo '?'))"
    fi

    if [[ $INSTALL_CUDA -eq 1 && $HAVE_NVCC -eq 0 ]]; then
        info "Installing CUDA toolkit ${CUDA_VERSION} (via zypper) ..."
        # NVIDIA publishes a single 'opensuse15' repo for all Leap 15.x/SLES 15.x.
        # zypper addrepo needs the bare repository directory URL (not a .repo file).
        local ARCH="x86_64"
        local REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/opensuse15/${ARCH}"
        local ALIAS="cuda-nvidia"
        # Remove any stale aliases from previous runs (both old and current name).
        # Use both zypper removerepo and direct file deletion: removerepo can
        # silently fail when the repo metadata is already broken/inconsistent.
        $SUDO zypper --non-interactive removerepo cuda        2>/dev/null || true
        $SUDO zypper --non-interactive removerepo cuda-nvidia 2>/dev/null || true
        $SUDO rm -f /etc/zypp/repos.d/cuda.repo \
                    /etc/zypp/repos.d/cuda-nvidia.repo 2>/dev/null || true
        $SUDO zypper --non-interactive addrepo --refresh "$REPO_URL" "$ALIAS"
        $SUDO zypper --non-interactive --gpg-auto-import-keys refresh "$ALIAS"
        $SUDO zypper --non-interactive install "cuda-toolkit-${CUDA_PKG_VER}"
        ok "CUDA toolkit installed. Add /usr/local/cuda/bin to PATH before building:"
        info "  export PATH=/usr/local/cuda/bin:\$PATH"
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
    local arch_pkgs=()
    [[ $HAVE_CMAKE -eq 0 ]] && arch_pkgs+=(cmake)
    [[ $HAVE_NINJA -eq 0 ]] && arch_pkgs+=(ninja)
    [[ ${#arch_pkgs[@]} -gt 0 ]] && $SUDO pacman -S --noconfirm --needed "${arch_pkgs[@]}"

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

    [[ $HAVE_CMAKE -eq 0 ]] && brew install cmake
    [[ $HAVE_NINJA -eq 0 ]] && brew install ninja

    if [[ $INSTALL_CUDA -eq 1 ]]; then
        warn "CUDA is not supported on macOS (Apple Silicon). Skipping."
    fi
}

# ---- main ------------------------------------------------------------------
detect_os
check_existing

case "$OS" in
    debian)   install_debian ;;
    fedora)   install_fedora ;;
    opensuse) install_opensuse ;;
    arch)     install_arch ;;
    macos)    install_macos ;;
    *)        fail "Unsupported OS. Please install a C++17 compiler manually." ;;
esac

echo ""
ok "Build dependencies installed successfully."
if [[ $INSTALL_CUDA -eq 1 && $HAVE_NVCC -eq 0 ]]; then
    info "Remember to add CUDA to your PATH:"
    info "  export PATH=/usr/local/cuda/bin:\$PATH"
fi
info "You can now build pygrog with:  pip install -e ."
