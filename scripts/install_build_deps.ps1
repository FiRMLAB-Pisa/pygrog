<# .SYNOPSIS
    Install C++ (and optionally CUDA) build toolchain on Windows.
.DESCRIPTION
    Installs Visual Studio Build Tools with C++ workload and optionally
    the NVIDIA CUDA Toolkit.  Skips components already present.

    Requires Administrator privileges.
.PARAMETER Cuda
    Install CUDA toolkit alongside the C++ toolchain.
.PARAMETER CudaVersion
    CUDA toolkit version to install (default: 12.6.0).
.EXAMPLE
    .\scripts\install_build_deps.ps1
    .\scripts\install_build_deps.ps1 -Cuda
#>
[CmdletBinding()]
param(
    [switch]$Cuda,
    [string]$CudaVersion = "12.6.0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---- helpers ---------------------------------------------------------------
function Write-Info  { Write-Host "[INFO]  $args" -ForegroundColor Cyan }
function Write-Ok    { Write-Host "[OK]    $args" -ForegroundColor Green }
function Write-Warn  { Write-Host "[WARN]  $args" -ForegroundColor Yellow }
function Write-Fail  { Write-Host "[FAIL]  $args" -ForegroundColor Red; exit 1 }

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]$identity
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# ---- check privileges ------------------------------------------------------
if (-not (Test-Admin)) {
    Write-Fail "This script must be run as Administrator.`nRight-click PowerShell -> Run as Administrator."
}

# ---- check existing tools --------------------------------------------------
Write-Info "Checking existing tools..."

$haveCL = $false
$haveNvcc = $false

# cl.exe (MSVC)
if (Test-Command "cl") {
    Write-Ok "cl.exe found"
    $haveCL = $true
} else {
    # Check common VS paths
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $vsPath = & $vsWhere -latest -property installationPath 2>$null
        if ($vsPath) {
            Write-Ok "Visual Studio found at $vsPath"
            $haveCL = $true
        }
    }
    if (-not $haveCL) {
        Write-Warn "No C++ compiler (cl.exe / Visual Studio) found"
    }
}

# nvcc
if (Test-Command "nvcc") {
    $nvccOut = nvcc --version 2>&1 | Select-String 'release'
    Write-Ok "nvcc found: $nvccOut"
    $haveNvcc = $true
} else {
    if ($Cuda) { Write-Warn "nvcc not found" }
}

# ---- install Visual Studio Build Tools -------------------------------------
if (-not $haveCL) {
    Write-Info "Installing Visual Studio Build Tools with C++ workload..."
    $vsUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"
    $installer = "$env:TEMP\vs_buildtools.exe"

    Write-Info "Downloading Build Tools installer..."
    Invoke-WebRequest -Uri $vsUrl -OutFile $installer -UseBasicParsing

    Write-Info "Running installer (this may take a while)..."
    $proc = Start-Process -FilePath $installer -ArgumentList @(
        "--quiet", "--wait", "--norestart",
        "--add", "Microsoft.VisualStudio.Workload.VCTools",
        "--includeRecommended"
    ) -Wait -PassThru

    if ($proc.ExitCode -eq 0 -or $proc.ExitCode -eq 3010) {
        Write-Ok "Visual Studio Build Tools installed (may require reboot)."
    } else {
        Write-Fail "VS Build Tools installer exited with code $($proc.ExitCode)."
    }

    Remove-Item $installer -Force -ErrorAction SilentlyContinue
} else {
    Write-Ok "C++ compiler already present"
}

# ---- install CUDA Toolkit --------------------------------------------------
if ($Cuda -and -not $haveNvcc) {
    Write-Info "Installing CUDA Toolkit $CudaVersion ..."
    $cudaMajorMinor = ($CudaVersion -split '\.')[0..1] -join '.'
    $cudaUrl = "https://developer.download.nvidia.com/compute/cuda/$CudaVersion/local_installers/cuda_${CudaVersion}_windows.exe"
    $cudaInstaller = "$env:TEMP\cuda_installer.exe"

    Write-Info "Downloading CUDA installer (this is a large download)..."
    Invoke-WebRequest -Uri $cudaUrl -OutFile $cudaInstaller -UseBasicParsing

    Write-Info "Running CUDA installer (silent, toolkit only)..."
    $proc = Start-Process -FilePath $cudaInstaller -ArgumentList @(
        "-s", "cuda_toolkit_${CudaVersion}"
    ) -Wait -PassThru

    if ($proc.ExitCode -eq 0) {
        Write-Ok "CUDA Toolkit $CudaVersion installed."
    } else {
        Write-Warn "CUDA installer exited with code $($proc.ExitCode). You may need to install manually."
    }

    Remove-Item $cudaInstaller -Force -ErrorAction SilentlyContinue
} elseif ($Cuda) {
    Write-Ok "CUDA (nvcc) already present"
}

# ---- summary ---------------------------------------------------------------
Write-Host ""
Write-Ok "Build dependencies installed successfully."
Write-Info "You can now build pygrog with:  pip install -e ."
if (-not $haveCL) {
    Write-Info "You may need to restart your terminal or reboot for VS paths to take effect."
}
