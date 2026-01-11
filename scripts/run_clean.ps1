# Run extraction for CLEAN sources (No subtitles)
$InputPath = "D:\Projects\Training\No Sub"
$OutputPath = "D:\Projects\Training\Triplets"
$TempDir = "D:\Projects\Training\Temp"

Write-Host "Starting extraction for CLEAN sources..."
Write-Host "Input: $InputPath"
Write-Host "Output: $OutputPath"

# Create temp directory if needed
if (-not (Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

# No crop argument needed
python scripts/extract_triplets.py --input $InputPath --output $OutputPath --temp-dir $TempDir

Write-Host "Clean extraction complete!"
Pause
