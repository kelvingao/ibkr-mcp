"""
Setup script for building Cython extensions.

This script compiles protected modules to .so/.pyd files,
making the source code difficult to reverse engineer.

Build commands:
    python setup.py build_ext --inplace   # Build for local testing
    python -m build --wheel               # Build wheel for distribution
"""

from pathlib import Path

from setuptools import setup
from setuptools.extension import Extension

# Check if Cython is available
try:
    from Cython.Build import cythonize

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("Warning: Cython not found. Building without compilation.")

# Base directory
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"

# Modules to compile (protect source code)
# These are the core business logic files
PROTECTED_MODULES = [
    # Core tools and server logic
    "ibkr_mcp/tools.py",
    # Common utilities with business logic
    "ibkr_mcp/common/greeks.py",
    "ibkr_mcp/common/rules.py",
    "ibkr_mcp/common/playbooks.py",
    "ibkr_mcp/common/positions.py",
    "ibkr_mcp/common/option_data.py",
    # All strategies (high value)
    "ibkr_mcp/strategies/base.py",
    "ibkr_mcp/strategies/strategy_covered_call.py",
    "ibkr_mcp/strategies/strategy_iron_condor.py",
    "ibkr_mcp/strategies/strategy_pmcc.py",
    "ibkr_mcp/strategies/strategy_put_credit_spread.py",
    "ibkr_mcp/strategies/strategy_vertical_spread.py",
    # Services
    "ibkr_mcp/services/base.py",
    "ibkr_mcp/services/account_service.py",
    "ibkr_mcp/services/news_service.py",
    "ibkr_mcp/services/option_data_service.py",
    "ibkr_mcp/services/risk_service.py",
]


def get_extensions():
    """Create Extension objects for all protected modules."""
    extensions = []

    for module_path in PROTECTED_MODULES:
        full_path = SRC_DIR / module_path
        if not full_path.exists():
            print(f"Warning: {full_path} not found, skipping...")
            continue

        # Convert path to module name: ibkr_mcp/tools.py -> ibkr_mcp.tools
        module_name = module_path.replace("/", ".").replace(".py", "")

        extensions.append(
            Extension(
                module_name,
                sources=[str(full_path)],
                extra_compile_args=["-O3"],  # Optimization level
            )
        )

    return extensions


def build_extensions():
    """Build Cython extensions if available."""
    if not USE_CYTHON:
        return []

    extensions = get_extensions()
    if not extensions:
        return []

    return cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
        build_dir="build",
        annotate=False,  # Set to True for HTML annotation during debug
    )


# Only run setup if this is the main script
if __name__ == "__main__":
    ext_modules = build_extensions()

    setup(
        ext_modules=ext_modules,
    )
