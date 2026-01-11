# Run extraction for ALL sources - both clean and hard subbed
# This script is designed to be scheduled to run overnight

$CleanPath = "C:\Projects\Training\No Sub"
$HardsubPath = "C:\Projects\Training\Hard sub"
$OutputPath = "E:\Triplets"
$TempDir = "C:\Projects\Training\Temp"
$ScriptDir = "C:\Projects\AInimotion"

# Log file for tracking progress
$LogFile = "E:\Triplets\extraction_log.txt"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Tee-Object -FilePath $LogFile -Append
}

Write-Log "========================================="
Write-Log "Starting overnight extraction job"
Write-Log "========================================="

# Create temp directory if needed
if (-not (Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

# Create output directory if needed
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
}

Set-Location $ScriptDir

# Process CLEAN sources first (no crop)
Write-Log ""
Write-Log "Processing CLEAN sources from: $CleanPath"
python scripts/extract_triplets.py --input $CleanPath --output $OutputPath --temp-dir $TempDir 2>&1 | Tee-Object -FilePath $LogFile -Append

# Process HARD SUBBED sources (with top+bottom crop)
Write-Log ""
Write-Log "Processing HARD SUBBED sources from: $HardsubPath"
Write-Log "Applying 150px top crop + 100px bottom crop"
python scripts/extract_triplets.py --input $HardsubPath --output $OutputPath --crop-top 150 --crop-bottom 100 --temp-dir $TempDir 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Log ""
Write-Log "========================================="
Write-Log "Extraction complete!"
Write-Log "Output: $OutputPath"

# Count results
$tripletCount = (Get-ChildItem $OutputPath -Directory).Count
Write-Log "Total triplet folders: $tripletCount"
Write-Log "========================================="

# Clean up temp
Remove-Item "$TempDir\*" -Recurse -Force -ErrorAction SilentlyContinue
