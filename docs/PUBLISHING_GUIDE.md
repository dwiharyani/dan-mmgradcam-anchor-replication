# Publishing the curated release to GitHub

## On your Mac

1. Download and unzip the GitHub-ready archive, and open a Terminal **inside the resulting folder**.
2. Preview the exact selected scripts and figures. This is **not** the entire HPC experiment workspace.
3. `python3 tools/audit_repository.py` (checks for accidental data/model/secret-like files).
4. `git init -b main`
5. `git add README.md .gitignore .gitattributes requirements.txt LICENSE_NOTICE.md docs/ figures/ scripts/ tools/ .github/`
6. `git status --short` and `git diff --cached --stat` (manually review). Do **not** use `git add -f`.
7. `python3 tools/audit_repository.py --all`
8. `git commit -m "Release independent MM-GradCAM replication and DAN Module 1 exploratory analysis"`
9. In GitHub, create a **new empty public repository**, for example `dan-mmgradcam-anchor-replication`. Do not initialize it with a README, `.gitignore`, or license; these already exist locally.
10. Copy your own GitHub account name into:

```bash
git remote add origin https://github.com/YOUR_USERNAME/dan-mmgradcam-anchor-replication.git
git push -u origin main
```

Authenticate via GitHub's normal browser/credential manager or a supported token/SSH method; never embed tokens in a script or README. The README images render automatically on GitHub. If Git says `remote origin already exists`, inspect it with `git remote -v` rather than resetting blindly.

## About model-training source

The shared release contains **all verified local post-training scripts** used in alignment, exact pairing, anchor-style visual reproduction, Module 1, sensitivity and visualization. The upstream author code is **linked**, not republished. If you want to release separately modified model-training runners from your HPC `src/` directory, review their code ownership/license and any secrets; add only the actual files you have inspected, and explain their origin. Do not recursively copy HPC `src/` or any `data/`, `results/`, `splits/`, model weights, CAM arrays, or other binary artifacts.
