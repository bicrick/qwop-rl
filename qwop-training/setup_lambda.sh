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
echo -e "${YELLOW}[1/6] Checking for Conda installation...${NC}"
if ! command -v conda &> /dev/null; then
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
else
    echo -e "${GREEN}Conda already installed${NC}"
    eval "$(conda shell.bash hook)"
fi
echo ""

# Step 2: Create/Update Conda Environment
echo -e "${YELLOW}[2/6] Setting up Conda environment 'qwop'...${NC}"
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
echo -e "${YELLOW}[3/6] Checking for Chrome-based browser...${NC}"
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
            sudo apt-get update
            sudo apt-get install -y chromium-browser
            CHROME_PATH=$(which chromium-browser)
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

# Step 4: Download ChromeDriver
echo -e "${YELLOW}[4/6] Downloading ChromeDriver...${NC}"

# Get Chrome version
if [[ "$OS" == "Linux" ]]; then
    CHROME_VERSION=$($CHROME_PATH --version | grep -oP '\d+\.\d+\.\d+' | head -1)
elif [[ "$OS" == "Darwin" ]]; then
    CHROME_VERSION=$("$CHROME_PATH" --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
fi

CHROME_MAJOR_VERSION=$(echo $CHROME_VERSION | cut -d. -f1)
echo "Chrome version: $CHROME_VERSION (major: $CHROME_MAJOR_VERSION)"

# Determine platform for ChromeDriver
if [[ "$OS" == "Linux" ]]; then
    PLATFORM="linux64"
elif [[ "$OS" == "Darwin" ]]; then
    if [[ "$ARCH" == "arm64" ]]; then
        PLATFORM="mac-arm64"
    else
        PLATFORM="mac-x64"
    fi
fi

# Download ChromeDriver
CHROMEDRIVER_URL="https://storage.googleapis.com/chrome-for-testing-public/$CHROME_VERSION/$PLATFORM/chromedriver-$PLATFORM.zip"

echo "Downloading ChromeDriver from: $CHROMEDRIVER_URL"
if wget -q --spider "$CHROMEDRIVER_URL" 2>/dev/null; then
    wget -q --show-progress "$CHROMEDRIVER_URL" -O chromedriver.zip
    unzip -q -o chromedriver.zip
    
    # Move chromedriver to project root
    if [[ -d "chromedriver-$PLATFORM" ]]; then
        mv "chromedriver-$PLATFORM/chromedriver" ./chromedriver
        rm -rf "chromedriver-$PLATFORM"
    fi
    rm chromedriver.zip
    chmod +x chromedriver
    echo -e "${GREEN}ChromeDriver downloaded successfully${NC}"
else
    echo -e "${YELLOW}Exact version not found, trying latest stable...${NC}"
    # Fallback to latest stable
    LATEST_URL="https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"
    wget -q "$LATEST_URL" -O versions.json
    
    # This is a simplified approach - you may need jq for better parsing
    echo -e "${YELLOW}Please download ChromeDriver manually from:${NC}"
    echo "https://googlechromelabs.github.io/chrome-for-testing/"
    rm -f versions.json
fi

CHROMEDRIVER_PATH="$(pwd)/chromedriver"
echo ""

# Step 5: Install Python dependencies
echo -e "${YELLOW}[5/6] Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q qwop-gym

echo -e "${GREEN}Python dependencies installed${NC}"
echo ""

# Step 6: Patch QWOP source code
echo -e "${YELLOW}[6/6] Patching QWOP source code...${NC}"
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
echo "To activate the environment, run:"
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
