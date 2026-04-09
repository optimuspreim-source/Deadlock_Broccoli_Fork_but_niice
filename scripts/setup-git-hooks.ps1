param(
    [switch]$LocalOnly = $true
)

$ErrorActionPreference = "Stop"

if ($LocalOnly) {
    git config --local core.hooksPath .githooks
    git config --local commit.template .gitmessage.txt
    git config --local commit.cleanup strip
    Write-Host "Configured local git settings for this repository."
} else {
    git config --global core.hooksPath .githooks
    git config --global commit.template .gitmessage.txt
    git config --global commit.cleanup strip
    Write-Host "Configured global git settings (affects all repositories)."
}

Write-Host 'Done. Test with: git commit -m "feat(core): test commit format" --allow-empty'