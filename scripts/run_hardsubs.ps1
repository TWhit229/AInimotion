# Run extraction for HARD SUBBED sources (Crops top and bottom text)
$InputPath = "D:\Projects\Training\Hard sub"
$OutputPath = "D:\Projects\Training\Triplets"
$TempDir = "D:\Projects\Training\Temp"

Write-Host "Starting extraction for HARD SUBBED sources..."
Write-Host "Input: $InputPath"
Write-Host "Output: $OutputPath"
Write-Host "Applying 150px top crop + 100px bottom crop to remove all subtitles..."

# Create temp directory if needed
if (-not (Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

# Apply both crop-top and crop-bottom (subtitles appear at both locations)
python scripts/extract_triplets.py --input $InputPath --output $OutputPath --crop-top 150 --crop-bottom 100 --temp-dir $TempDir

Write-Host "Hardsub extraction complete!"
Pause
