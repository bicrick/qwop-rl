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
echo -e "${YELLOW}[4/6] Checking for ChromeDriver...${NC}"

CHROMEDRIVER_PATH="$(pwd)/chromedriver"
CHROMEDRIVER_DOWNLOADED=false

# Check if ChromeDriver already exists
if [[ -f "$CHROMEDRIVER_PATH" ]]; then
    echo -e "${GREEN}ChromeDriver already exists at: $CHROMEDRIVER_PATH${NC}"
else
    echo "ChromeDriver not found. Setting up..."
    
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
        if [[ "$ARCH" == "aarch64" ]]; then
            # For aarch64, try google's platform first, but it likely doesn't exist
            PLATFORM="linux64"
        elif [[ "$ARCH" == "x86_64" ]]; then
            PLATFORM="linux64"
        else
            echo -e "${RED}Unsupported Linux architecture for ChromeDriver: $ARCH${NC}"
            exit 1
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        if [[ "$ARCH" == "arm64" ]]; then
            PLATFORM="mac-arm64"
        else
            PLATFORM="mac-x64"
        fi
    fi
    
    # Try to download ChromeDriver from Google's official source
    echo "Attempting to download ChromeDriver from Google Chrome for Testing..."
    CHROMEDRIVER_URL="https://storage.googleapis.com/chrome-for-testing-public/$CHROME_VERSION/$PLATFORM/chromedriver-$PLATFORM.zip"
    
    if wget -q --spider "$CHROMEDRIVER_URL" 2>/dev/null; then
        echo "Downloading: $CHROMEDRIVER_URL"
        if wget -q --show-progress "$CHROMEDRIVER_URL" -O chromedriver.zip; then
            if unzip -q -o chromedriver.zip; then
                # Move chromedriver to project root
                if [[ -d "chromedriver-$PLATFORM" ]]; then
                    mv "chromedriver-$PLATFORM/chromedriver" ./chromedriver
                    rm -rf "chromedriver-$PLATFORM"
                fi
                rm -f chromedriver.zip
                chmod +x chromedriver
                echo -e "${GREEN}ChromeDriver downloaded successfully${NC}"
                CHROMEDRIVER_DOWNLOADED=true
            fi
        fi
    else
        echo -e "${YELLOW}Exact version not found at Google source, trying alternative methods...${NC}"
    fi
    
    # Fallback for aarch64: use ChromeDriver from Chromium snap if available
    if [[ "$CHROMEDRIVER_DOWNLOADED" == false ]] && [[ "$OS" == "Linux" ]] && [[ "$ARCH" == "aarch64" ]]; then
        echo "Attempting to use ChromeDriver from Chromium snap (for aarch64)..."
        if [[ -f "/snap/chromium/current/usr/lib/chromium-browser/chromedriver" ]]; then
            echo "Found ChromeDriver in snap, copying..."
            cp /snap/chromium/current/usr/lib/chromium-browser/chromedriver ./chromedriver
            chmod +x chromedriver
            echo -e "${GREEN}ChromeDriver copied from snap successfully${NC}"
            CHROMEDRIVER_DOWNLOADED=true
        elif [[ -f "/usr/bin/chromedriver" ]]; then
            echo "Found ChromeDriver in /usr/bin, copying..."
            cp /usr/bin/chromedriver ./chromedriver
            chmod +x chromedriver
            echo -e "${GREEN}ChromeDriver copied from /usr/bin successfully${NC}"
            CHROMEDRIVER_DOWNLOADED=true
        fi
    fi
    
    # Try full version with patch number if direct download failed
    if [[ "$CHROMEDRIVER_DOWNLOADED" == false ]]; then
        echo -e "${YELLOW}Trying to find full version string with patch number...${NC}"
        LATEST_URL="https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"
        if wget -q "$LATEST_URL" -O versions.json 2>/dev/null; then
            if command -v jq &> /dev/null; then
                FULL_VERSION=$(jq -r ".versions[] | select(.version | startswith(\"$CHROME_MAJOR_VERSION.\")) | .version" versions.json | head -1)
                if [[ ! -z "$FULL_VERSION" ]]; then
                    echo "Found full version: $FULL_VERSION"
                    CHROMEDRIVER_URL="https://storage.googleapis.com/chrome-for-testing-public/$FULL_VERSION/$PLATFORM/chromedriver-$PLATFORM.zip"
                    echo "Trying: $CHROMEDRIVER_URL"
                    
                    if wget -q --spider "$CHROMEDRIVER_URL" 2>/dev/null; then
                        wget -q --show-progress "$CHROMEDRIVER_URL" -O chromedriver.zip
                        if unzip -q -o chromedriver.zip; then
                            if [[ -d "chromedriver-$PLATFORM" ]]; then
                                mv "chromedriver-$PLATFORM/chromedriver" ./chromedriver
                                rm -rf "chromedriver-$PLATFORM"
                            fi
                            rm -f chromedriver.zip
                            chmod +x chromedriver
                            echo -e "${GREEN}ChromeDriver downloaded successfully${NC}"
                            CHROMEDRIVER_DOWNLOADED=true
                        fi
                    fi
                fi
            fi
            rm -f versions.json
        fi
    fi
    
    # Final result
    if [[ "$CHROMEDRIVER_DOWNLOADED" == false ]]; then
        if [[ ! -f "$CHROMEDRIVER_PATH" ]]; then
            echo -e "${RED}ERROR: Could not obtain ChromeDriver${NC}"
            echo "Please download manually from: https://googlechromelabs.github.io/chrome-for-testing/"
            echo "Select ChromeDriver version $CHROME_VERSION for $PLATFORM"
            echo "Extract and place in: $CHROMEDRIVER_PATH"
            exit 1
        fi
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
