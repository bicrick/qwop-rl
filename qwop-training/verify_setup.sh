#!/bin/bash
# Verification script to test QWOP training setup

set -e

echo "=========================================="
echo "QWOP Setup Verification"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

# Check 1: Conda installation
echo -n "Checking Conda... "
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version 2>&1)
    echo -e "${GREEN}✓${NC} $CONDA_VERSION"
else
    echo -e "${RED}✗ Not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check 2: Conda environment
echo -n "Checking 'qwop' conda environment... "
if conda env list 2>/dev/null | grep -q "^qwop "; then
    echo -e "${GREEN}✓${NC} Exists"
    
    # Check Python version in environment
    PYTHON_VERSION=$(conda run -n qwop python --version 2>&1)
    echo "  Python version: $PYTHON_VERSION"
    
    if [[ ! "$PYTHON_VERSION" =~ "3.10" ]]; then
        echo -e "  ${YELLOW}⚠${NC} Python 3.10 recommended, found: $PYTHON_VERSION"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${RED}✗ Not found${NC}"
    echo "  Run: conda create -n qwop python=3.10 -y"
    ERRORS=$((ERRORS + 1))
fi

# Check 3: Chromium
echo -n "Checking Chromium browser... "
if command -v chromium-browser &> /dev/null; then
    CHROME_PATH=$(which chromium-browser)
    CHROME_VERSION=$(chromium-browser --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
    echo -e "${GREEN}✓${NC} Found at $CHROME_PATH"
    echo "  Version: $CHROME_VERSION"
elif command -v chromium &> /dev/null; then
    CHROME_PATH=$(which chromium)
    CHROME_VERSION=$(chromium --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
    echo -e "${GREEN}✓${NC} Found at $CHROME_PATH"
    echo "  Version: $CHROME_VERSION"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "  Run: sudo apt-get install chromium-browser"
    ERRORS=$((ERRORS + 1))
    CHROME_VERSION="unknown"
fi

# Check 4: ChromeDriver
echo -n "Checking ChromeDriver... "
CHROMEDRIVER_FOUND=false

# Check local directory first
if [[ -f "./chromedriver" ]]; then
    DRIVER_VERSION=$(./chromedriver --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
    echo -e "${GREEN}✓${NC} Found in current directory"
    echo "  Version: $DRIVER_VERSION"
    CHROMEDRIVER_FOUND=true
elif command -v chromedriver &> /dev/null; then
    DRIVER_PATH=$(which chromedriver)
    DRIVER_VERSION=$(chromedriver --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1)
    echo -e "${GREEN}✓${NC} Found at $DRIVER_PATH"
    echo "  Version: $DRIVER_VERSION"
    CHROMEDRIVER_FOUND=true
else
    echo -e "${RED}✗ Not found${NC}"
    echo "  Run: sudo apt-get install chromium-chromedriver"
    ERRORS=$((ERRORS + 1))
    DRIVER_VERSION="unknown"
fi

# Check version compatibility
if [[ "$CHROMEDRIVER_FOUND" == true ]] && [[ "$CHROME_VERSION" != "unknown" ]] && [[ "$DRIVER_VERSION" != "unknown" ]]; then
    CHROME_MAJOR=$(echo $CHROME_VERSION | cut -d. -f1)
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
    
    echo -n "Checking version compatibility... "
    if [[ "$CHROME_MAJOR" == "$DRIVER_MAJOR" ]]; then
        echo -e "${GREEN}✓${NC} Major versions match ($CHROME_MAJOR)"
    else
        echo -e "${RED}✗${NC} Version mismatch!"
        echo "  Chromium major: $CHROME_MAJOR"
        echo "  ChromeDriver major: $DRIVER_MAJOR"
        echo "  Fix: sudo apt-get install --reinstall chromium-browser chromium-chromedriver"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check 5: Xvfb
echo -n "Checking Xvfb (virtual display)... "
if command -v Xvfb &> /dev/null; then
    echo -e "${GREEN}✓${NC} Installed"
else
    echo -e "${YELLOW}⚠${NC} Not found (needed for headless training)"
    echo "  Run: sudo apt-get install xvfb"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 6: qwop-gym package
echo -n "Checking qwop-gym package... "
if conda run -n qwop python -c "import qwop_gym" 2>/dev/null; then
    QWOP_VERSION=$(conda run -n qwop python -c "import qwop_gym; print(qwop_gym.__version__)" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓${NC} Installed (version: $QWOP_VERSION)"
else
    echo -e "${RED}✗ Not installed${NC}"
    echo "  Run: conda activate qwop && pip install qwop-gym"
    ERRORS=$((ERRORS + 1))
fi

# Check 7: QWOP patched files
echo -n "Checking QWOP patched files... "
QWOP_DIR="$HOME/.qwop-gym"
if [[ -d "$QWOP_DIR" ]] && [[ -f "$QWOP_DIR/QWOP.min.js" ]]; then
    echo -e "${GREEN}✓${NC} Found at $QWOP_DIR"
else
    echo -e "${YELLOW}⚠${NC} Not found"
    echo "  Run: curl -sL https://www.foddy.net/QWOP.min.js | qwop-gym patch"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 8: Config files
echo -n "Checking config files... "
if [[ -f "config/env.yml" ]]; then
    echo -e "${GREEN}✓${NC} config/env.yml exists"
else
    echo -e "${YELLOW}⚠${NC} config/env.yml not found"
    echo "  Will be created by setup script"
    WARNINGS=$((WARNINGS + 1))
fi

# Check 9: Architecture info
echo ""
echo "System Information:"
echo "  OS: $(uname -s)"
echo "  Architecture: $(uname -m)"
if [[ "$(uname -m)" == "aarch64" ]]; then
    echo -e "  ${GREEN}✓${NC} ARM64 detected - using distro Chromium/ChromeDriver"
fi

# Summary
echo ""
echo "=========================================="
if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}All checks passed!${NC}"
    echo ""
    echo "You're ready to train:"
    echo "  conda activate qwop"
    echo "  ./lambda_train.sh train_ppo"
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}Setup complete with $WARNINGS warning(s)${NC}"
    echo ""
    echo "You can proceed, but consider fixing warnings above."
else
    echo -e "${RED}Setup incomplete: $ERRORS error(s), $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before training."
    exit 1
fi
echo "=========================================="
