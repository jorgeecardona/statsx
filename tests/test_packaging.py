import subprocess
import sys
from importlib.metadata import requires

FORBIDDEN_ON_IMPORT = (
    "torch",
    "pandas",
    "matplotlib",
    "scipy",
    "jax",
    "sqlalchemy",
    "sqlite3",
    "requests",
    "urllib3",
    "httpx",
    "boto3",
    "google.cloud",
)


def test_numpy_is_the_only_run_time_dependency() -> None:
    run_time = [r for r in requires("statsx") or [] if "extra ==" not in r]
    names = {r.split(";")[0].split("[")[0].split(">")[0].split("=")[0].strip() for r in run_time}
    assert names <= {"numpy"}


def test_import_loads_no_heavy_or_network_modules() -> None:
    code = "import sys, statsx; print(' '.join(sys.modules))"
    loaded = subprocess.run(  # noqa: S603 (fixed argv: this interpreter and a literal)
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    bad = [m for m in loaded if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_ON_IMPORT)]
    assert bad == []
