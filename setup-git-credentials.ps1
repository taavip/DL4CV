param(
  [string]$Name = "Your Name",
  [string]$Email = "you@example.com"
)

Write-Host "Setting repository-scoped Git identity..."
git config user.name "$Name"
git config user.email "$Email"

Write-Host "Enabling Windows credential manager (global) to cache PATs..."
git config --global credential.helper manager-core

Write-Host "Verification:"
Write-Host "  repo user.name:"; git config --get user.name
Write-Host "  repo user.email:"; git config --get user.email
Write-Host "  global credential.helper:"; git config --get --global credential.helper

Write-Host "
Next steps:"
Write-Host "  1) Create a GitHub Personal Access Token (PAT) with 'repo' scope at https://github.com/settings/tokens"
Write-Host "  2) Run: git push origin main — when prompted use your GitHub username and PAT as the password."
Write-Host "  3) To use different identity, re-run this script with -Name and -Email parameters."
