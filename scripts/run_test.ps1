# Test script - Run extraction on single video to verify it works
$InputFile = "C:\Projects\Training\No Sub\[Judas] Kimi no Na Wa. (Your Name.) [BD 2160p 4K UHD][HEVC x265 10bit][Dual-Audio][Multi-Subs].mkv"
$OutputPath = "E:\Triplets"
$TempDir = "C:\Projects\Training\Temp"

Write-Host "========================================="
Write-Host "TEST RUN - Single Video Extraction"
Write-Host "========================================="
Write-Host "Input: $InputFile"
Write-Host "Output: $OutputPath"
Write-Host "Temp Dir: $TempDir"
Write-Host ""

# Check if file exists
if (-not (Test-Path $InputFile)) {
    Write-Host "ERROR: File not found!" -ForegroundColor Red
    Pause
    exit 1
}

# Create temp directory if needed
if (-not (Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

Write-Host "Starting extraction (this may take a while for 4K video)..."
Write-Host ""

# Run extraction - no crop since this is a clean source
Set-Location "C:\Projects\AInimotion"
python scripts/extract_triplets.py --input "$InputFile" --output $OutputPath --temp-dir $TempDir

Write-Host ""
Write-Host "========================================="
Write-Host "Test extraction complete!"
Write-Host "Check $OutputPath for triplet folders"
Write-Host "========================================="
Pause
