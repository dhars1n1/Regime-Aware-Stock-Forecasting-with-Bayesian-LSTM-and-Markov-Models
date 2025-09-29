# ============================================================================
# Windows Long Path Fix & TensorFlow Installation Script
# ============================================================================

Write-Host "🔧 FIXING WINDOWS LONG PATH ISSUE & INSTALLING TENSORFLOW" -ForegroundColor Yellow
Write-Host "=" * 70

# Step 1: Enable Long Path Support (Registry Fix)
Write-Host "`n1️⃣ ENABLING WINDOWS LONG PATH SUPPORT..." -ForegroundColor Cyan
Write-Host "   This requires Administrator privileges..."

try {
    # Check if running as administrator
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if (-not $isAdmin) {
        Write-Host "   ⚠️  This script needs to run as Administrator to enable Long Path support!" -ForegroundColor Red
        Write-Host "   📋 MANUAL STEPS (Run as Administrator):" -ForegroundColor Yellow
        Write-Host "   1. Right-click PowerShell -> Run as Administrator"
        Write-Host "   2. Run: New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force"
        Write-Host "   3. Restart your computer"
        Write-Host ""
        Write-Host "   🔄 ALTERNATIVE: Use Group Policy (gpedit.msc)"
        Write-Host "   Computer Configuration -> Administrative Templates -> System -> Filesystem"
        Write-Host "   Enable 'Enable Win32 long paths'"
        Write-Host ""
    } else {
        # Enable Long Path support via registry
        New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
        Write-Host "   ✅ Long Path support enabled in registry!" -ForegroundColor Green
        Write-Host "   ⚠️  Restart required for full effect, but we can try installation now" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ❌ Could not modify registry: $($_.Exception.Message)" -ForegroundColor Red
}

# Step 2: Clean Virtual Environment
Write-Host "`n2️⃣ CLEANING VIRTUAL ENVIRONMENT..." -ForegroundColor Cyan

# Activate virtual environment
$venvPath = ".\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "   🔄 Activating virtual environment..."
    & $venvPath
    
    Write-Host "   🧹 Removing corrupted TensorFlow packages..."
    pip uninstall tensorflow tensorflow-gpu tensorflow-cpu -y
    
    Write-Host "   🗑️ Clearing pip cache..."
    pip cache purge
    
    Write-Host "   ✅ Virtual environment cleaned!" -ForegroundColor Green
} else {
    Write-Host "   ❌ Virtual environment not found at $venvPath" -ForegroundColor Red
    Write-Host "   📋 Creating new virtual environment..."
    python -m venv venv
    & ".\venv\Scripts\Activate.ps1"
}

# Step 3: Install TensorFlow with workarounds
Write-Host "`n3️⃣ INSTALLING TENSORFLOW WITH WORKAROUNDS..." -ForegroundColor Cyan

# Upgrade pip first
Write-Host "   📦 Upgrading pip..."
python -m pip install --upgrade pip

# Method 1: Try CPU-only version first (more reliable)
Write-Host "   🎯 Attempting TensorFlow CPU installation..."
try {
    pip install tensorflow-cpu==2.16.1 --no-cache-dir
    Write-Host "   ✅ TensorFlow CPU installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️ CPU version failed, trying regular TensorFlow..." -ForegroundColor Yellow
    
    # Method 2: Try with different installation options
    Write-Host "   🔄 Trying with --no-deps and shorter paths..."
    pip install tensorflow==2.16.1 --no-cache-dir --no-deps
    pip install tensorflow==2.16.1 --no-cache-dir --force-reinstall
}

# Step 4: Test Installation
Write-Host "`n4️⃣ TESTING TENSORFLOW INSTALLATION..." -ForegroundColor Cyan
python -c "
try:
    import tensorflow as tf
    print('   ✅ TensorFlow imported successfully!')
    print(f'   📋 Version: {tf.__version__}')
    print(f'   🖥️ Available devices: {len(tf.config.list_physical_devices())}')
    print('   🎉 Installation SUCCESSFUL!')
except ImportError as e:
    print('   ❌ TensorFlow import failed:', str(e))
    print('   🔄 Trying alternative solutions...')
"

Write-Host "`n🎯 POST-INSTALLATION STEPS:" -ForegroundColor Yellow
Write-Host "=" * 40
Write-Host "1. If TensorFlow works -> Run: python run_complete_pipeline.py"
Write-Host "2. If still failing -> Restart computer and try again"
Write-Host "3. Alternative -> Use conda instead of pip"
Write-Host ""
Write-Host "🆘 BACKUP PLAN - CONDA INSTALLATION:"
Write-Host "   conda create -n tf_env python=3.11"
Write-Host "   conda activate tf_env"  
Write-Host "   conda install tensorflow"
Write-Host ""
Write-Host "✅ SCRIPT COMPLETE!" -ForegroundColor Green