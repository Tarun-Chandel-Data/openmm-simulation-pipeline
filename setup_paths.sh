#!/bin/bash
# ---- setup_paths.sh ----
# Run this once, from inside the project folder, after cloning the repo.
# Replaces the __PROJECT_DIR__ placeholder with your actual current path
# in every .py, .in, .inp, and .sh file.

NEWPATH=$(pwd)
echo "Setting all pipeline paths to: $NEWPATH"

find . -type f \( -name "*.py" -o -name "*.in" -o -name "*.inp" -o -name "*.sh" \) \
  -exec sed -i "s|__PROJECT_DIR__|$NEWPATH|g" {} +

echo "Done. Verifying no placeholders remain:"
grep -rl "__PROJECT_DIR__" --include="*.py" --include="*.in" --include="*.inp" --include="*.sh" . \
  && echo "WARNING: some files still contain the placeholder (see above)" \
  || echo "All files updated successfully."
