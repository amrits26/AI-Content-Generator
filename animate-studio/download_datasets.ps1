# ============================================
# AniMate Studio Dataset Downloader
# ============================================

$ErrorActionPreference = 'Stop'

function Log($msg) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
}

# --- Colibri Dataset ---
$colibriDir = "datasets/colibri"
$colibriZip = "$colibriDir/ColibriImagesDataset.zip"
$colibriCSV = "$colibriDir/ColibriImagesMetadataAnnotations.csv"
$colibriURL = "https://zenodo.org/records/15535927/files/ColibriImagesDataset.zip"
$colibriCSVURL = "https://zenodo.org/records/15535927/files/ColibriImagesMetadataAnnotations.csv"

Log "Creating Colibri dataset directory..."
New-Item -ItemType Directory -Path $colibriDir -Force | Out-Null

if (!(Test-Path $colibriZip)) {
    Log "Downloading Colibri ZIP..."
    Invoke-WebRequest -Uri $colibriURL -OutFile $colibriZip -Resume
} else {
    Log "Colibri ZIP already exists."
}

if (!(Test-Path $colibriCSV)) {
    Log "Downloading Colibri metadata CSV..."
    Invoke-WebRequest -Uri $colibriCSVURL -OutFile $colibriCSV -Resume
} else {
    Log "Colibri CSV already exists."
}

Log "Extracting Colibri ZIP..."
Expand-Archive -Path $colibriZip -DestinationPath $colibriDir -Force

# --- Ot & Sien Dataset ---
$otSienDir = "datasets/ot_sien"
New-Item -ItemType Directory -Path $otSienDir -Force | Out-Null
$otSienURL = "https://lab.kb.nl/dataset/ot-sien-dataset"

Log "Checking for Ot & Sien direct download..."
# No direct download link; provide manual instructions
Log "Manual download required for Ot & Sien. Please visit: $otSienURL"
Log "After downloading, extract images and annotations to $otSienDir."

# --- Verify Files ---
if (!(Test-Path $colibriZip) -or !(Test-Path $colibriCSV)) {
    Log "❌ Colibri files missing. Please check your internet connection."
    exit 1
}

Log "✅ Download complete."
