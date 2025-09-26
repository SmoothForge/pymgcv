#!/bin/bash
cd "$PIXI_PROJECT_ROOT"
find . -name "*.ipynb" -print0 | while IFS= read -r -d '' nb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb" --ExecutePreprocessor.kernel_name=pymgcv --ClearMetadataPreprocessor.enabled=True
done
