param(
    [ValidateSet("all", "run_game", "run_red", "run_blue", "clean_logs")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$GameExe = Join-Path $Root "Build\TreasureHunt.exe"
$RedClient = Join-Path $Root "testClientRed.py"
$BlueClient = Join-Path $Root "testClientBlue.py"

function Start-Game {
    Start-Process -FilePath $GameExe
}

function Start-RedClient {
    Start-Process -FilePath $RedClient
}

function Start-BlueClient {
    Start-Process -FilePath $BlueClient
}

function Clean-Logs {
    Remove-Item -Path `
        (Join-Path $Root "log_red.txt"), `
        (Join-Path $Root "log_blue.txt") `
        -Force `
        -ErrorAction SilentlyContinue
}

switch ($Target) {
    "run_game" {
        Start-Game
    }
    "run_red" {
        Start-RedClient
    }
    "run_blue" {
        Start-BlueClient
    }
    "clean_logs" {
        Clean-Logs
    }
    "all" {
        Clean-Logs
        Start-Game
        Start-Sleep -Seconds 4
        Start-RedClient
        Start-Sleep -Seconds 2
        Start-BlueClient
    }
}
