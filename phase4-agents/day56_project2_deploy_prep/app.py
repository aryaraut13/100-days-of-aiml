import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def check_environment() -> bool:
    """Verify all required environment variables are set."""
    required = ["ANTHROPIC_API_KEY"]
    missing  = [var for var in required if not os.getenv(var)]

    if missing:
        logger.error(f"Missing environment variables: {missing}")
        return False

    logger.info("Environment check passed")
    return True


def check_dependencies() -> bool:
    """Verify all required packages are importable."""
    required_packages = [
        "langchain_anthropic",
        "langgraph",
        "streamlit",
        "dotenv",
    ]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(f"Missing packages: {missing}")
        return False

    logger.info("Dependency check passed")
    return True


def health_check() -> dict:
    """Run all health checks and return status."""
    checks = {
        "environment": check_environment(),
        "dependencies": check_dependencies(),
    }
    checks["overall"] = all(checks.values())
    return checks


if __name__ == "__main__":
    print("[PROJECT 2 — DEPLOYMENT READINESS CHECK]\n")

    status = health_check()

    print(f"Environment vars:  {'OK' if status['environment']  else 'FAIL'}")
    print(f"Dependencies:      {'OK' if status['dependencies'] else 'FAIL'}")
    print(f"\nOverall status:    {'READY TO DEPLOY' if status['overall'] else 'NOT READY'}")

    if not status["overall"]:
        print("\nFix the issues above before deploying.")
        sys.exit(1)
    else:
        print("\nAll checks passed. Ready for deployment.")