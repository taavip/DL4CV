This repo uses an HTTPS remote. Commits previously failed because `user.name` and `user.email` were not set locally.

Steps (you chose repository-scoped identity + HTTPS+PAT):

1) From the repository root, set repository identity (local only):

```powershell
# replace with your name/email
git config user.name "Your Name"
git config user.email "you@example.com"
```

2) Enable Windows credential manager (global) so your PAT will be cached:

```powershell
git config --global credential.helper manager-core
```

3) Create a GitHub Personal Access Token (PAT):
- Visit https://github.com/settings/tokens -> Generate new token -> give it `repo` scope (and `workflow` if needed). Copy the token.

4) Push to origin:

```powershell
git push origin main
```

When prompted, enter your GitHub username and use the PAT as the password. Windows Credential Manager will store it.

Optional: Use SSH instead (alternative):

```powershell
ssh-keygen -t ed25519 -C "you@example.com"
# then add the contents of ~/.ssh/id_ed25519.pub to GitHub > Settings > SSH and GPG keys
# update remote to SSH if you prefer:
git remote set-url origin git@github.com:taavip/DL4CV.git
```

Files created:
- `setup-git-credentials.ps1` — PowerShell helper script to set local identity and enable credential manager.
- `SETUP_GIT_CREDENTIALS.md` — step-by-step instructions.

If you want, I can: run the verification commands in your PowerShell terminal (I will not change anything without your permission), or modify the script to set global identity instead.