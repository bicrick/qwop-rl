#!/bin/bash
set -e

echo "=========================================="
echo "QWOP Gym Lambda Instance Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

echo -e "${GREEN}Detected OS: $OS${NC}"
echo -e "${GREEN}Detected Architecture: $ARCH${NC}"
echo ""

# Step 1: Install Miniconda if not present
echo -e "${YELLOW}[1/7] Checking for Conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    # Check if Miniconda directory already exists
    if [[ -d "$HOME/miniconda3" ]]; then
        echo -e "${YELLOW}Miniconda directory found at $HOME/miniconda3. Initializing...${NC}"
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        conda init bash
        echo -e "${GREEN}Miniconda initialized${NC}"
    else
        echo "Conda not found. Installing Miniconda..."
        
        if [[ "$OS" == "Linux" ]]; then
            if [[ "$ARCH" == "x86_64" ]]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            elif [[ "$ARCH" == "aarch64" ]]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
            else
                echo -e "${RED}Unsupported architecture: $ARCH${NC}"
                exit 1
            fi
        elif [[ "$OS" == "Darwin" ]]; then
            if [[ "$ARCH" == "arm64" ]]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
            else
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            fi
        else
            echo -e "${RED}Unsupported OS: $OS${NC}"
            exit 1
        fi
        
        wget -q --show-progress "$MINICONDA_URL" -O miniconda_installer.sh
        bash miniconda_installer.sh -b -p "$HOME/miniconda3"
        rm miniconda_installer.sh
        
        # Initialize conda
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
        conda init bash
        
        echo -e "${GREEN}Miniconda installed successfully${NC}"
    fi
else
    echo -e "${GREEN}Conda already installed${NC}"
    eval "$(conda shell.bash hook)"
fi
echo ""

# Step 2: Create/Update Conda Environment
echo -e "${YELLOW}[2/7] Setting up Conda environment 'qwop'...${NC}"
if conda env list | grep -q "^qwop "; then
    echo "Environment 'qwop' already exists. Updating..."
    conda activate qwop
else
    echo "Creating new environment 'qwop' with Python 3.10..."
    conda create -n qwop python=3.10 -y
    conda activate qwop
fi
echo -e "${GREEN}Conda environment ready${NC}"
echo ""

# Step 3: Install Chrome/Chromium
echo -e "${YELLOW}[3/7] Checking for Chrome-based browser...${NC}"
CHROME_PATH=""

if [[ "$OS" == "Linux" ]]; then
    # Check for common Chrome installations on Linux
    if command -v google-chrome &> /dev/null; then
        CHROME_PATH=$(which google-chrome)
    elif command -v chromium-browser &> /dev/null; then
        CHROME_PATH=$(which chromium-browser)
    elif command -v chromium &> /dev/null; then
        CHROME_PATH=$(which chromium)
    else
        echo "Chrome not found. Installing Chromium..."
        if command -v apt-get &> /dev/null; then
            # Debian/Ubuntu
            echo "Installing Chromium and ChromeDriver together..."
            sudo apt-get update -qq
            
            # Install both chromium-browser and chromium-chromedriver together
            # This ensures version compatibility (critical for ARM64)
            if [[ "$ARCH" == "aarch64" ]]; then
                echo "Detected ARM64 architecture."
                echo "Note: Using distro-provided Chromium (no official Google Chrome for linux-arm64)"
                sudo apt-get install -y chromium-browser chromium-chromedriver
                CHROME_PATH=$(which chromium-browser 2>/dev/null || which chromium)
            else
                # For x86_64, also use apt for consistency
                sudo apt-get install -y chromium-browser chromium-chromedriver
                CHROME_PATH=$(which chromium-browser 2>/dev/null || which chromium)
            fi
        elif command -v yum &> /dev/null; then
            # RHEL/CentOS
            sudo yum install -y chromium
            CHROME_PATH=$(which chromium)
        else
            echo -e "${RED}Could not install Chrome automatically. Please install manually.${NC}"
            exit 1
        fi
    fi
elif [[ "$OS" == "Darwin" ]]; then
    # macOS
    if [ -f "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]; then
        CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    elif [ -f "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" ]; then
        CHROME_PATH="/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    elif [ -f "/Applications/Chromium.app/Contents/MacOS/Chromium" ]; then
        CHROME_PATH="/Applications/Chromium.app/Contents/MacOS/Chromium"
    else
        echo -e "${RED}No Chrome-based browser found. Please install Google Chrome, Brave, or Chromium.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Chrome found at: $CHROME_PATH${NC}"
echo ""

# Step 4: Setup ChromeDriver
echo -e "${YELLOW}[4/7] Setting up ChromeDriver...${NC}"

CHROMEDRIVER_PATH="$(pwd)/chromedriver"
CHROMEDRIVER_FOUND=false

# Check if ChromeDriver already exists in current directory
if [[ -f "$CHROMEDRIVER_PATH" ]]; then
    echo -e "${GREEN}ChromeDriver already exists at: $CHROMEDRIVER_PATH${NC}"
    CHROMEDRIVER_FOUND=true
else
    echo "ChromeDriver not found in current directory. Checking system..."
    
    # Get Chrome/Chromium version
    if [[ "$OS" == "Linux" ]]; then
        CHROME_VERSION=$($CHROME_PATH --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "unknown")
    elif [[ "$OS" == "Darwin" ]]; then
        CHROME_VERSION=$("$CHROME_PATH" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
    fi
    
    if [[ "$CHROME_VERSION" != "unknown" ]]; then
        echo "Chromium version: $CHROME_VERSION"
    fi
    
    # For Linux (both x86_64 and ARM64), check for system chromedriver
    if [[ "$OS" == "Linux" ]]; then
        if command -v chromedriver &> /dev/null; then
            SYSTEM_DRIVER=$(which chromedriver)
            DRIVER_VERSION=$(chromedriver --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "unknown")
            echo "Found system ChromeDriver: $SYSTEM_DRIVER"
            echo "ChromeDriver version: $DRIVER_VERSION"
            
            # Copy to local directory for consistency
            cp "$SYSTEM_DRIVER" ./chromedriver
            chmod +x chromedriver
            echo -e "${GREEN}ChromeDriver copied from system${NC}"
            CHROMEDRIVER_FOUND=true
            
            # Verify version compatibility
            if [[ "$CHROME_VERSION" != "unknown" ]] && [[ "$DRIVER_VERSION" != "unknown" ]]; then
                CHROME_MAJOR=$(echo $CHROME_VERSION | cut -d. -f1)
                DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
                
                if [[ "$CHROME_MAJOR" != "$DRIVER_MAJOR" ]]; then
                    echo -e "${YELLOW}WARNING: Version mismatch!${NC}"
                    echo "  Chromium major version: $CHROME_MAJOR"
                    echo "  ChromeDriver major version: $DRIVER_MAJOR"
                    echo "  This may cause issues. Consider reinstalling both packages together."
                else
                    echo -e "${GREEN}Version check: Chromium and ChromeDriver major versions match${NC}"
                fi
            fi
        else
            echo -e "${YELLOW}ChromeDriver not found in system${NC}"
            
            # For ARM64, we must use system package (no official Google builds)
            if [[ "$ARCH" == "aarch64" ]]; then
                echo "ARM64 detected - ChromeDriver must be installed via apt"
                echo "Installing chromium-chromedriver..."
                sudo apt-get update -qq
                sudo apt-get install -y chromium-chromedriver
                
                if command -v chromedriver &> /dev/null; then
                    SYSTEM_DRIVER=$(which chromedriver)
                    cp "$SYSTEM_DRIVER" ./chromedriver
                    chmod +x chromedriver
                    echo -e "${GREEN}ChromeDriver installed and copied${NC}"
                    CHROMEDRIVER_FOUND=true
                fi
            fi
        fi
    fi
    
    # For macOS, try downloading official ChromeDriver from Google
    if [[ "$CHROMEDRIVER_FOUND" == false ]] && [[ "$OS" == "Darwin" ]]; then
        if [[ "$CHROME_VERSION" != "unknown" ]]; then
            if [[ "$ARCH" == "arm64" ]]; then
                PLATFORM="mac-arm64"
            else
                PLATFORM="mac-x64"
            fi
            
            echo "Downloading ChromeDriver for macOS..."
            CHROMEDRIVER_URL="https://storage.googleapis.com/chrome-for-testing-public/$CHROME_VERSION/$PLATFORM/chromedriver-$PLATFORM.zip"
            
            if wget -q --spider "$CHROMEDRIVER_URL" 2>/dev/null; then
                wget -q --show-progress "$CHROMEDRIVER_URL" -O chromedriver.zip
                unzip -q -o chromedriver.zip
                if [[ -d "chromedriver-$PLATFORM" ]]; then
                    mv "chromedriver-$PLATFORM/chromedriver" ./chromedriver
                    rm -rf "chromedriver-$PLATFORM"
                fi
                rm -f chromedriver.zip
                chmod +x chromedriver
                echo -e "${GREEN}ChromeDriver downloaded${NC}"
                CHROMEDRIVER_FOUND=true
            fi
        fi
    fi
    
    # Final check
    if [[ "$CHROMEDRIVER_FOUND" == false ]]; then
        echo -e "${RED}ERROR: Could not setup ChromeDriver${NC}"
        echo ""
        if [[ "$ARCH" == "aarch64" ]]; then
            echo "For ARM64 Ubuntu, both Chromium and ChromeDriver should be installed together:"
            echo "  sudo apt-get install chromium-browser chromium-chromedriver"
            echo ""
            echo "This ensures version compatibility (critical on ARM64)."
        else
            echo "Please ensure chromium-chromedriver is installed:"
            echo "  sudo apt-get install chromium-chromedriver"
        fi
        exit 1
    fi
fi

# Verify chromedriver works
if [[ -f "$CHROMEDRIVER_PATH" ]]; then
    if ./chromedriver --version &>/dev/null; then
        FINAL_VERSION=$(./chromedriver --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1 || echo "unknown")
        echo -e "${GREEN}ChromeDriver is ready: $FINAL_VERSION${NC}"
    else
        echo -e "${YELLOW}Warning: ChromeDriver exists but may not be executable${NC}"
    fi
fi
echo ""

# Step 5: Install Xvfb for headless training
echo -e "${YELLOW}[5/7] Checking for Xvfb (virtual display)...${NC}"
if command -v Xvfb &> /dev/null; then
    echo -e "${GREEN}Xvfb already installed${NC}"
else
    echo "Installing Xvfb for headless training..."
    if [[ "$OS" == "Linux" ]]; then
        if command -v apt-get &> /dev/null; then
            # Debian/Ubuntu
            sudo apt-get update -qq
            sudo apt-get install -y xvfb
        elif command -v yum &> /dev/null; then
            # RHEL/CentOS
            sudo yum install -y xorg-x11-server-Xvfb
        fi
    fi
    echo -e "${GREEN}Xvfb installed${NC}"
fi
echo ""

# Step 6: Install Python dependencies
echo -e "${YELLOW}[6/7] Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q qwop-gym

echo -e "${GREEN}Python dependencies installed${NC}"
echo ""

# Step 7: Patch QWOP source code
echo -e "${YELLOW}[7/7] Patching QWOP source code...${NC}"
curl -sL https://www.foddy.net/QWOP.min.js | qwop-gym patch
echo -e "${GREEN}QWOP source code patched${NC}"
echo ""

# Create config directory if it doesn't exist
mkdir -p config

# Create env.yml config file
echo -e "${YELLOW}Creating configuration file...${NC}"
cat > config/env.yml << EOF
browser: "$CHROME_PATH"
driver: "$CHROMEDRIVER_PATH"
EOF

echo -e "${GREEN}Configuration saved to config/env.yml${NC}"
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "To use conda in this shell session, run:"
echo "  source ~/.bashrc"
echo "  # OR"
echo "  eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\""
echo ""
echo "Then activate the environment:"
echo "  conda activate qwop"
echo ""
echo "To test the installation, run:"
echo "  qwop-gym play"
echo ""
echo "Configuration:"
echo "  Browser: $CHROME_PATH"
echo "  ChromeDriver: $CHROMEDRIVER_PATH"
echo "  Config file: config/env.yml"
echo ""
echo "For training and other commands, see README.md"
echo ""
