# Read-only host inventory. Docker is not started by this script.
$ErrorActionPreference = 'Continue'
$rootPath = Split-Path -Parent $PSScriptRoot
$inventory = [ordered]@{
    timestamp_utc = [DateTime]::UtcNow.ToString('o')
    os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, TotalVisibleMemorySize, FreePhysicalMemory
    drives = @(Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root, Used, Free)
    commands = @(Get-Command python*, py, pip, uv, conda, docker, nvcc, nvidia-smi -All -ErrorAction SilentlyContinue | Select-Object Name, Source)
    environment = @(Get-ChildItem Env: | Where-Object Name -Match '^(VIRTUAL_ENV|CONDA_PREFIX|CONDA_DEFAULT_ENV)$')
    wsl = ((wsl --list --verbose 2>&1 | Out-String) -replace "`0", '')
    docker_containers = (docker ps -a 2>&1 | Out-String)
    docker_images = (docker images 2>&1 | Out-String)
    gpu = (nvidia-smi 2>&1 | Out-String)
}
$inventory | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $rootPath 'audit/stage_a_windows_inventory.json')
