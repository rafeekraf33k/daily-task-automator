@echo off
title Clean up G:\antigrvty
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = Get-Content -LiteralPath '%~f0' -Raw; $m = '#PS' + '_START'; Invoke-Expression $s.Substring($s.IndexOf($m) + $m.Length)"
echo.
pause
exit /b

#PS_START
$Path = "G:\antigrvty"
$ErrorActionPreference = "Stop"

$categories = [ordered]@{
    "Images"        = "jpg jpeg png gif bmp webp tif tiff svg ico heic raw psd ai"
    "Videos"        = "mp4 mkv avi mov wmv flv webm m4v 3gp mpg mpeg"
    "Audio"         = "mp3 wav flac aac ogg m4a wma opus"
    "PDFs"          = "pdf"
    "Documents"     = "doc docx txt rtf odt md epub"
    "Spreadsheets"  = "xls xlsx xlsm csv ods"
    "Presentations" = "ppt pptx odp key"
    "Archives"      = "zip rar 7z tar gz bz2 xz iso"
    "Installers"    = "exe msi apk dmg msix appx"
    "Code"          = "py js ts html htm css json xml yaml yml ps1 bat sh java c cpp h cs php sql ipynb mq4 mq5 ex4 ex5"
    "Fonts"         = "ttf otf woff woff2"
}
$lookup = @{}
foreach ($cat in $categories.Keys) {
    foreach ($ext in $categories[$cat].Split(" ")) { $lookup[".$ext"] = $cat }
}

if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
    Write-Host ""
    Write-Host "  Can't find $Path" -ForegroundColor Red
    Write-Host "  Check the drive is plugged in and the folder name is right."
    return
}
$logFile = Join-Path $Path "_sort_log.csv"

function Get-FreeName([string]$dir, [string]$name) {
    $target = Join-Path $dir $name
    if (-not (Test-Path -LiteralPath $target)) { return $target }
    $base = [IO.Path]::GetFileNameWithoutExtension($name)
    $ext  = [IO.Path]::GetExtension($name)
    $i = 1
    do {
        $target = Join-Path $dir ("{0} ({1}){2}" -f $base, $i, $ext)
        $i++
    } while (Test-Path -LiteralPath $target)
    return $target
}

$files = @(Get-ChildItem -LiteralPath $Path -File -Force | Where-Object {
    $_.Name -notin @("_sort_log.csv", "desktop.ini", "Thumbs.db", "Clean_antigrvty.bat") -and
    -not ($_.Attributes -band [IO.FileAttributes]::System)
})

# ---------- nothing to sort: offer undo
if ($files.Count -eq 0) {
    Write-Host ""
    Write-Host "  All clean! No loose files in $Path" -ForegroundColor Green
    if (Test-Path -LiteralPath $logFile) {
        Write-Host ""
        Write-Host "  Want to UNDO the last clean-up? Press U. Any other key = exit."
        $k = [Console]::ReadKey($true).KeyChar
        if ("$k".ToUpper() -eq "U") {
            $rows = @(Import-Csv -LiteralPath $logFile)
            [array]::Reverse($rows)
            foreach ($r in $rows) {
                if (Test-Path -LiteralPath $r.To) {
                    Move-Item -LiteralPath $r.To -Destination (Get-FreeName (Split-Path $r.From) (Split-Path $r.From -Leaf))
                }
            }
            Remove-Item -LiteralPath $logFile
            foreach ($cat in @($categories.Keys) + "Other") {
                $d = Join-Path $Path $cat
                if ((Test-Path -LiteralPath $d) -and -not (Get-ChildItem -LiteralPath $d -Force)) { Remove-Item -LiteralPath $d }
            }
            Write-Host "  Undone. Everything is back where it was." -ForegroundColor Green
        }
    }
    return
}

# ---------- show the plan
$plan = foreach ($f in $files) {
    $cat = $lookup[$f.Extension.ToLower()]
    if (-not $cat) { $cat = "Other" }
    [pscustomobject]@{ File = $f; Category = $cat }
}
Write-Host ""
Write-Host "  Found $($files.Count) loose files in $Path" -ForegroundColor Cyan
Write-Host "  They will go into these folders:"
Write-Host ""
$plan | Group-Object Category | Sort-Object Count -Descending | ForEach-Object {
    Write-Host ("     {0,-14} {1,5} files" -f $_.Name, $_.Count)
}
Write-Host ""
Write-Host "  Your existing folders are NOT touched. Nothing is deleted." -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Press Y to clean up now. Any other key = cancel." -ForegroundColor Yellow
$k = [Console]::ReadKey($true).KeyChar
if ("$k".ToUpper() -ne "Y") { Write-Host "  Cancelled. Nothing moved."; return }

# ---------- move
$log = New-Object System.Collections.Generic.List[object]
$failed = 0
foreach ($p in $plan) {
    $dir = Join-Path $Path $p.Category
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $dest = Get-FreeName $dir $p.File.Name
    try {
        Move-Item -LiteralPath $p.File.FullName -Destination $dest
        $log.Add([pscustomobject]@{ From = $p.File.FullName; To = $dest })
    } catch {
        Write-Host "  Skipped $($p.File.Name) (it's probably open in another app)" -ForegroundColor Red
        $failed++
    }
}
if ($log.Count) {
    $old = @()
    if (Test-Path -LiteralPath $logFile) { $old = @(Import-Csv -LiteralPath $logFile) }
    ($old + $log) | Export-Csv -LiteralPath $logFile -NoTypeInformation -Encoding UTF8
}
Write-Host ""
Write-Host "  Done! $($log.Count) files sorted." -ForegroundColor Green
if ($failed) { Write-Host "  $failed skipped - close those apps and double-click again." }
Write-Host "  Changed your mind? Double-click this file again and press U to undo."
Start-Process explorer.exe $Path
