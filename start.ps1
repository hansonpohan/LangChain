# LangChain RAG 系統 PowerShell 啟動腳本
# 適用於 Windows 系統

Write-Host "🚀 啟動 LangChain RAG 系統..." -ForegroundColor Green
Write-Host "=" * 50

# 檢查 Python 是否安裝
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python 已安裝: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python 未安裝，請先安裝 Python" -ForegroundColor Red
    exit 1
}

# 檢查虛擬環境
if (Test-Path "venv") {
    Write-Host "📦 激活虛擬環境..." -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "⚠️  未找到虛擬環境，建議創建虛擬環境：" -ForegroundColor Yellow
    Write-Host "python -m venv venv" -ForegroundColor Cyan
    Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "pip install -r requirements.txt" -ForegroundColor Cyan
}

# 檢查依賴
Write-Host "🔍 檢查依賴..." -ForegroundColor Yellow
$dependencies = @("flask", "langchain", "pinecone", "sentence_transformers", "google.generativeai")
$missingDeps = @()

foreach ($dep in $dependencies) {
    try {
        python -c "import $dep" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ $dep" -ForegroundColor Green
        } else {
            $missingDeps += $dep
        }
    } catch {
        $missingDeps += $dep
    }
}

if ($missingDeps.Count -gt 0) {
    Write-Host "❌ 缺少依賴: $($missingDeps -join ', ')" -ForegroundColor Red
    Write-Host "請運行: pip install -r requirements.txt" -ForegroundColor Yellow
    $install = Read-Host "是否現在安裝依賴? (y/N)"
    if ($install -eq "y" -or $install -eq "Y") {
        pip install -r requirements.txt
    } else {
        exit 1
    }
}

# 檢查環境文件
if (Test-Path ".env") {
    Write-Host "✅ 找到環境配置文件" -ForegroundColor Green
} else {
    Write-Host "⚠️  未找到 .env 文件，使用預設配置" -ForegroundColor Yellow
}

Write-Host "=" * 50
Write-Host "📊 系統信息:" -ForegroundColor Cyan
Write-Host "工作目錄: $(Get-Location)" -ForegroundColor White
Write-Host "時間: $(Get-Date)" -ForegroundColor White
Write-Host "=" * 50

# 啟動說明
Write-Host "🌐 啟動說明:" -ForegroundColor Cyan
Write-Host "  後端 API: http://localhost:5001" -ForegroundColor White
Write-Host "  前端界面: 請用瀏覽器打開 frontend.html" -ForegroundColor White
Write-Host "  按 Ctrl+C 停止服務器" -ForegroundColor White
Write-Host "=" * 50

# 檢查端口是否被佔用
$port = 5001
try {
    $connection = Test-NetConnection -ComputerName localhost -Port $port -InformationLevel Quiet -WarningAction SilentlyContinue
    if ($connection) {
        Write-Host "⚠️  端口 $port 已被佔用" -ForegroundColor Yellow
    }
} catch {
    # 忽略錯誤，繼續執行
}

try {
    # 啟動應用
    Write-Host "🚀 啟動 Flask 應用..." -ForegroundColor Green
    python app_langchain.py
} catch {
    Write-Host "❌ 啟動失敗: $_" -ForegroundColor Red
    exit 1
}

Write-Host "👋 服務器已停止" -ForegroundColor Yellow
