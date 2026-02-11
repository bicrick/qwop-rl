#!/bin/bash
# Verification script to check if QWOP Gym is properly set up

echo "=========================================="
echo "QWOP Gym Setup Verification"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# 1. Check Conda
echo "Checking Conda installation..."
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version)
    check_pass "Conda found: $CONDA_VERSION"
else
    check_fail "Conda not found"
fi
echo ""

# 2. Check Conda environment
echo "Checking Conda environment 'qwop'..."
if conda env list | grep -q "^qwop "; then
    check_pass "Environment 'qwop' exists"
    
    # Check if activated
    if [[ "$CONDA_DEFAULT_ENV" == "qwop" ]]; then
        check_pass "Environment 'qwop' is activated"
    else
        check_warn "Environment 'qwop' is not activated. Run: conda activate qwop"
    fi
else
    check_fail "Environment 'qwop' not found"
fi
echo ""

# 3. Check Python version (if env is activated)
if [[ "$CONDA_DEFAULT_ENV" == "qwop" ]]; then
    echo "Checking Python version..."
    PYTHON_VERSION=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    if [[ "$PYTHON_VERSION" == "3.10" ]]; then
        check_pass "Python 3.10 found"
    else
        check_warn "Python version is $PYTHON_VERSION (expected 3.10)"
    fi
    echo ""
fi

# 4. Check qwop-gym package
echo "Checking qwop-gym package..."
if [[ "$CONDA_DEFAULT_ENV" == "qwop" ]]; then
    if python -c "import qwop_gym" 2>/dev/null; then
        QWOP_VERSION=$(python -c "import qwop_gym; print(qwop_gym.__version__)" 2>/dev/null || echo "unknown")
        check_pass "qwop-gym package installed (version: $QWOP_VERSION)"
    else
        check_fail "qwop-gym package not found. Run: pip install qwop-gym"
    fi
else
    check_warn "Activate 'qwop' environment to check package"
fi
echo ""

# 5. Check Chrome/Chromium
echo "Checking Chrome-based browser..."
CHROME_FOUND=false
if command -v google-chrome &> /dev/null; then
    CHROME_PATH=$(which google-chrome)
    CHROME_VERSION=$(google-chrome --version 2>/dev/null || echo "unknown")
    check_pass "Google Chrome found: $CHROME_VERSION"
    CHROME_FOUND=true
elif command -v chromium-browser &> /dev/null; then
    CHROME_PATH=$(which chromium-browser)
    CHROME_VERSION=$(chromium-browser --version 2>/dev/null || echo "unknown")
    check_pass "Chromium found: $CHROME_VERSION"
    CHROME_FOUND=true
elif command -v chromium &> /dev/null; then
    CHROME_PATH=$(which chromium)
    CHROME_VERSION=$(chromium --version 2>/dev/null || echo "unknown")
    check_pass "Chromium found: $CHROME_VERSION"
    CHROME_FOUND=true
elif [ -f "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]; then
    CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    CHROME_VERSION=$("$CHROME_PATH" --version 2>/dev/null || echo "unknown")
    check_pass "Google Chrome found: $CHROME_VERSION"
    CHROME_FOUND=true
elif [ -f "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" ]; then
    CHROME_PATH="/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    check_pass "Brave Browser found"
    CHROME_FOUND=true
else
    check_fail "No Chrome-based browser found"
fi
echo ""

# 6. Check ChromeDriver
echo "Checking ChromeDriver..."
if [ -f "./chromedriver" ]; then
    if [ -x "./chromedriver" ]; then
        DRIVER_VERSION=$(./chromedriver --version 2>/dev/null | head -1 || echo "unknown")
        check_pass "ChromeDriver found: $DRIVER_VERSION"
    else
        check_fail "ChromeDriver found but not executable. Run: chmod +x chromedriver"
    fi
else
    check_fail "ChromeDriver not found in current directory"
fi
echo ""

# 7. Check Xvfb (for headless systems)
echo "Checking Xvfb (for headless training)..."
if command -v Xvfb &> /dev/null; then
    check_pass "Xvfb found (required for headless training)"
else
    check_warn "Xvfb not found. Install with: sudo apt-get install xvfb"
fi
echo ""

# 8. Check config file
echo "Checking configuration..."
if [ -f "config/env.yml" ]; then
    check_pass "Configuration file found: config/env.yml"
    
    # Check if paths are set
    if grep -q "browser:" config/env.yml && grep -q "driver:" config/env.yml; then
        check_pass "Browser and driver paths configured"
    else
        check_warn "Configuration may be incomplete"
    fi
else
    check_warn "Configuration file not found: config/env.yml"
    echo "         Run setup_lambda.sh or qwop-gym bootstrap to create it"
fi
echo ""

# 9. Check QWOP patch
echo "Checking QWOP patch..."
if [[ "$CONDA_DEFAULT_ENV" == "qwop" ]]; then
    QWOP_FILE=$(python -c "import qwop_gym; import os; print(os.path.join(os.path.dirname(qwop_gym.__file__), 'envs/v1/game/QWOP.min.js'))" 2>/dev/null)
    if [ -f "$QWOP_FILE" ]; then
        check_pass "QWOP.min.js found and patched"
    else
        check_fail "QWOP.min.js not found. Run: curl -sL https://www.foddy.net/QWOP.min.js | qwop-gym patch"
    fi
else
    check_warn "Activate 'qwop' environment to check QWOP patch"
fi
echo ""

# Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Setup is complete.${NC}"
    echo ""
    echo "You can now run:"
    echo "  qwop-gym play          # Play the game"
    echo "  qwop-gym train_ppo     # Start training"
    echo "  ./lambda_train.sh      # Train with virtual display (Lambda)"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Setup is mostly complete with $WARNINGS warning(s).${NC}"
    echo "Review the warnings above and fix if needed."
else
    echo -e "${RED}Setup incomplete: $ERRORS error(s), $WARNINGS warning(s).${NC}"
    echo "Please fix the errors above before proceeding."
    exit 1
fi
echo ""
