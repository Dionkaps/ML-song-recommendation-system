param(
    [Parameter(Mandatory = $true)]
    [string] $SourceDir,

    [int] $TargetCount = 10000,

    [string[]] $Extensions = @(".mp3", ".wav", ".flac", ".m4a")
)

$ErrorActionPreference = "Stop"

$scriptPath = $MyInvocation.MyCommand.Path
$benchmarkRoot = Split-Path -Parent $scriptPath
$targetDir = Join-Path $benchmarkRoot "audio_10000"
$dataDir = Join-Path $benchmarkRoot "data"
$manifestPath = Join-Path $dataDir "copied_audio_manifest.csv"

$resolvedBenchmark = (Resolve-Path -LiteralPath $benchmarkRoot).Path
$resolvedTarget = if (Test-Path -LiteralPath $targetDir) {
    $targetItem = Get-Item -LiteralPath $targetDir
    if (($targetItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing target audio reparse point: $targetDir"
    }
    (Resolve-Path -LiteralPath $targetDir).Path
} else {
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    (Resolve-Path -LiteralPath $targetDir).Path
}

if ((Split-Path -Parent $resolvedTarget) -ne $resolvedBenchmark -or (Split-Path -Leaf $resolvedTarget) -ne "audio_10000") {
    throw "Refusing to copy to unexpected target: $resolvedTarget"
}

if (-not (Test-Path -LiteralPath $SourceDir)) {
    throw "Source directory does not exist: $SourceDir"
}

$resolvedSource = (Resolve-Path -LiteralPath $SourceDir).Path
if ($resolvedSource -eq $resolvedTarget -or $resolvedSource.StartsWith($resolvedBenchmark + [IO.Path]::DirectorySeparatorChar)) {
    throw "Source must be outside this benchmark folder: $resolvedSource"
}

$existingTargetFiles = @(Get-ChildItem -LiteralPath $resolvedTarget -File -Force)
if ($existingTargetFiles.Count -gt 0) {
    throw "Target audio_10000 is not empty. Refusing to mix benchmark inputs."
}

if (Test-Path -LiteralPath $dataDir) {
    $dataItem = Get-Item -LiteralPath $dataDir
    if (($dataItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing data directory reparse point: $dataDir"
    }
}

$normalizedExt = $Extensions | ForEach-Object { $_.ToLowerInvariant() }
$sourceFiles = @(
    Get-ChildItem -LiteralPath $resolvedSource -File -Recurse -Force |
        Where-Object { $normalizedExt -contains $_.Extension.ToLowerInvariant() } |
        Sort-Object FullName
)

if ($sourceFiles.Count -lt $TargetCount) {
    throw "Source has only $($sourceFiles.Count) audio files; need $TargetCount."
}

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$resolvedData = (Resolve-Path -LiteralPath $dataDir).Path
$resolvedManifestParent = Split-Path -Parent (Resolve-Path -LiteralPath $dataDir).Path
if ((Split-Path -Parent $resolvedData) -ne $resolvedBenchmark -or (Split-Path -Leaf $resolvedData) -ne "data") {
    throw "Refusing manifest data path outside benchmark root: $resolvedData"
}
$manifestFullPath = [IO.Path]::GetFullPath($manifestPath)
if (-not $manifestFullPath.StartsWith($resolvedData + [IO.Path]::DirectorySeparatorChar)) {
    throw "Refusing manifest path outside data directory: $manifestFullPath"
}

$selected = $sourceFiles | Select-Object -First $TargetCount
$manifestRows = New-Object System.Collections.Generic.List[object]

$i = 0
foreach ($file in $selected) {
    $i += 1
    $dest = Join-Path $resolvedTarget $file.Name
    if (Test-Path -LiteralPath $dest) {
        $base = [IO.Path]::GetFileNameWithoutExtension($file.Name)
        $ext = $file.Extension
        $dest = Join-Path $resolvedTarget ("{0}__copy{1:D5}{2}" -f $base, $i, $ext)
    }

    Copy-Item -LiteralPath $file.FullName -Destination $dest

    $manifestRows.Add([pscustomobject]@{
        Index = $i
        SourcePath = $file.FullName
        DestPath = (Resolve-Path -LiteralPath $dest).Path
        Bytes = $file.Length
    })
}

$manifestRows | Export-Csv -LiteralPath $manifestFullPath -NoTypeInformation -Encoding UTF8

[pscustomobject]@{
    Source = $resolvedSource
    Target = $resolvedTarget
    Copied = $manifestRows.Count
    Manifest = $manifestFullPath
}
