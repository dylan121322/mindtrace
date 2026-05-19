import os
import platform
import re
import shutil
import subprocess
from functools import lru_cache
from typing import List


def _run_command(args: List[str], timeout: int = 3) -> str:
    if not args:
        return ""
    exe = shutil.which(args[0]) if not os.path.isabs(args[0]) else args[0]
    if not exe:
        return ""
    try:
        result = subprocess.run(
            [exe, *args[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception:
        return ""
    return (result.stdout or "").strip()


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _powershell_lines(command: str) -> List[str]:
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        return []
    output = _run_command(
        [
            exe,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]
    )
    return _dedupe([line.strip() for line in output.splitlines()])


def _wmic_values(args: List[str], key: str) -> List[str]:
    output = _run_command(["wmic", *args], timeout=4)
    values: List[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower() == key.lower():
            continue
        if "=" in stripped:
            name, value = stripped.split("=", 1)
            if name.strip().lower() == key.lower():
                values.append(value)
        else:
            values.append(stripped)
    return _dedupe(values)


def _read_first_cpuinfo_value() -> str:
    cpuinfo = "/proc/cpuinfo"
    if not os.path.exists(cpuinfo):
        return ""
    try:
        with open(cpuinfo, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "hardware", "processor")) and ":" in line:
                    value = line.split(":", 1)[-1].strip()
                    if value:
                        return value
    except OSError:
        return ""
    return ""


def _detect_cpu_model() -> str:
    system = platform.system().lower()
    if system == "windows":
        lines = _powershell_lines("Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name")
        if lines:
            return lines[0]
        values = _wmic_values(["cpu", "get", "Name", "/value"], "Name")
        if values:
            return values[0]

    if system == "darwin":
        output = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=3)
        if output:
            return output.splitlines()[0].strip()

    if system == "linux":
        output = _run_command(["lscpu"], timeout=3)
        for line in output.splitlines():
            if line.lower().startswith("model name:"):
                return line.split(":", 1)[-1].strip()
        value = _read_first_cpuinfo_value()
        if value:
            return value

    return platform.processor() or platform.machine() or "Unknown CPU"


def _detect_windows_gpus() -> List[str]:
    lines = _powershell_lines("Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name")
    if lines:
        return lines
    return _wmic_values(["path", "win32_VideoController", "get", "Name", "/value"], "Name")


def _detect_macos_gpus() -> List[str]:
    output = _run_command(["system_profiler", "SPDisplaysDataType"], timeout=8)
    values: List[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model:"):
            values.append(stripped.split(":", 1)[-1])
    return _dedupe(values)


def _detect_linux_gpus() -> List[str]:
    values: List[str] = []
    nvidia = _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=4)
    values.extend([line.strip() for line in nvidia.splitlines() if line.strip()])

    rocm = _run_command(["rocm-smi", "--showproductname"], timeout=4)
    for line in rocm.splitlines():
        if "card series" in line.lower() or "product name" in line.lower():
            value = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
            values.append(value)

    lspci = _run_command(["lspci"], timeout=4)
    for line in lspci.splitlines():
        lower = line.lower()
        if "vga compatible controller" in lower or "3d controller" in lower or "display controller" in lower:
            values.append(re.sub(r"^[0-9a-f:.]+\s+", "", line, flags=re.I))
    return _dedupe(values)


def _detect_gpu_models() -> List[str]:
    system = platform.system().lower()
    if system == "windows":
        return _dedupe(_detect_windows_gpus())
    if system == "darwin":
        return _detect_macos_gpus()
    if system == "linux":
        return _detect_linux_gpus()
    return []


def _infer_accelerator(gpu_models: List[str]) -> str:
    joined = " ".join(gpu_models).lower()
    system = platform.system().lower()
    if "nvidia" in joined:
        return "cuda"
    if "amd" in joined or "radeon" in joined:
        return "rocm/directml"
    if system == "darwin" and gpu_models:
        return "metal"
    if "intel" in joined:
        return "directml/oneapi"
    return "cpu"


@lru_cache(maxsize=1)
def get_hardware_info() -> dict:
    cpu_count = os.cpu_count() or 1
    gpu_models = _detect_gpu_models()
    return {
        "platform": platform.system() or "Unknown",
        "machine": platform.machine() or "",
        "cpu_model": _detect_cpu_model(),
        "cpu_count": cpu_count,
        "recommended_workers": max(1, min(8, cpu_count)),
        "gpu_models": gpu_models,
        "accelerator": _infer_accelerator(gpu_models),
    }
