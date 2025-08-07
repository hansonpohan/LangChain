# LangChain RAG 系統

這是一個基於 LangChain 的檢索增強生成 (RAG) 系統，使用 Flask 作為後端 API，支援文檔上傳、文字處理和智能問答功能。

## 功能特點

- 🤖 **智能問答**: 基於上傳的文檔內容回答問題
- 📄 **多格式文檔支援**: 支援 PDF 和 TXT 文件上傳
- 📝 **文字直接上傳**: 支援直接貼上文字內容
- 🔍 **智能文檔分割**: 自動或自定義文檔分割策略
- 🎯 **向量檢索**: 使用 Pinecone 向量數據庫進行相似性搜索
- 🧠 **精確回答**: 集成 Google Gemini 2.5 Flash 生成準確的回答
- 🌐 **友好界面**: 現代化的 Web 前端界面
- 📊 **系統監控**: 實時查看系統狀態和統計資訊

## 技術架構

### 後端技術棧

- **LangChain**: 核心 RAG 框架
- **Flask**: Web 框架和 API 服務
- **Pinecone**: 向量數據庫 (1024 維)
- **Google Gemini 2.5 Flash**: 大語言模型
- **HuggingFace Embeddings**: sentence-transformers/all-roberta-large-v1
- **PyPDF2 & pdfplumber**: PDF 文件處理
- **RecursiveCharacterTextSplitter**: 智能文檔分割

### 前端技術棧

- **HTML5**: 結構
- **CSS3**: 樣式和動畫
- **JavaScript**: 交互邏輯
- **Fetch API**: HTTP 請求

## 系統要求

- **Python 版本**: 3.11.9 或更高版本
- **作業系統**: Windows 10/11, macOS, Linux
- **記憶體**: 建議 4GB 以上
- **磁盤空間**: 2GB 以上可用空間

## 安裝步驟

### 1. 確認 Python 版本

確保您的系統安裝了 Python 3.11.9：

```bash
python --version
# 應該顯示: Python 3.11.9 或更高版本
```

如果沒有安裝 Python 3.11.9，請從 [Python 官網](https://www.python.org/downloads/) 下載安裝。

### 2. 下載/克隆項目

```bash
git clone <repository-url>
cd LangChain
```

### 3. 創建虛擬環境 (建議)

```bash
# Windows
python -m venv langchainenv
langchainenv\Scripts\activate

# macOS/Linux
python -m venv langchainenv
source langchainenv/bin/activate
```

### 4. 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

### 5. 配置環境變量

創建 `.env` 文件並填入您的 API 金鑰：

```env
PINECONE_API_KEY=您的Pinecone_API金鑰
GEMINI_API_KEY=您的Google_Gemini_API金鑰
PINECONE_ENV=us-east-1
PINECONE_INDEX_NAME=langchain-rag-demo
```

### 6. 啟動應用

#### 方式一：使用 PowerShell 腳本 (Windows)

```powershell
.\start.ps1
```

#### 方式二：使用 Python 啟動腳本

```bash
python run.py
```

#### 方式三：直接啟動主應用

```bash
python app.py
```

### 7. 打開前端界面

啟動後訪問：

- 瀏覽器打開：`http://localhost:5001`
- 或直接打開 `frontend.html` 文件

## 使用指南

### 1. 文檔上傳

#### 文件上傳方式

1. 切換到「文檔上傳」標籤
2. 選擇「文件上傳」選項
3. 點擊「選擇文件」按鈕 (支援 PDF, TXT)
4. 填入文檔 ID（唯一標識符）
5. 填入來源描述（可選）
6. 調整分割策略（可選，系統會自動推薦）
7. 點擊「上傳文檔」

#### 文字上傳方式

1. 切換到「文檔上傳」標籤
2. 選擇「文字上傳」選項
3. 填入文檔 ID
4. 貼入文檔內容
5. 填入來源描述（可選）
6. 調整分割策略（可選）
7. 點擊「上傳文檔」

### 2. 智能問答

1. 切換到「智能問答」標籤
2. 輸入您的問題
3. 點擊「提交查詢」
4. 查看 AI 回答和相關的上下文信息
5. 查看來源文檔數量

### 3. 文檔搜索

1. 切換到「文檔搜索」標籤
2. 輸入搜索關鍵字
3. 設定返回結果數量 (預設 5)
4. 點擊「搜索文檔」
5. 查看相似文檔和相似度分數

### 4. 系統監控

1. 切換到「系統狀態」標籤
2. 點擊「檢查系統狀態」
3. 查看各組件運行狀態和統計信息
4. 監控 Pinecone 索引狀態

### 5. 分割策略推薦

1. 在上傳文檔前，可以使用「獲取分割建議」功能
2. 系統會根據文本長度自動推薦最佳分割參數
3. 支援自定義 chunk_size 和 chunk_overlap 參數

## API 接口

### GET /health

健康檢查，返回系統狀態

**回應範例:**

```json
{
  "status": "healthy",
  "components": {
    "embedding_model": "sentence-transformers/all-roberta-large-v1",
    "vector_store": "Pinecone",
    "llm": "Gemini-2.5-Flash",
    "qa_chain": "RetrievalQA"
  },
  "pinecone_stats": {
    "total_vector_count": 150,
    "dimension": 1024
  }
}
```

### POST /query

智能問答查詢

**請求格式:**

```json
{
  "query": "您的問題"
}
```

**回應格式:**

```json
{
  "success": true,
  "response": "AI 回答內容",
  "context": "相關上下文資訊",
  "source_documents_count": 3
}
```

### POST /upload

上傳文檔 (支援文字和文件)

#### 文字上傳

**請求格式:**

```json
{
  "id": "文檔ID",
  "text": "文檔內容",
  "source": "文檔來源",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

#### 文件上傳 (Multipart Form)

**表單欄位:**

- `file`: 文件 (PDF/TXT)
- `doc_id`: 文檔 ID
- `source`: 來源描述
- `chunk_size`: 分割大小 (可選)
- `chunk_overlap`: 重疊大小 (可選)

**回應格式:**

```json
{
  "success": true,
  "message": "文件已成功上傳...",
  "doc_id": "文檔ID",
  "content_length": 5000,
  "chunk_count": 5,
  "chunking_strategy": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
}
```

### POST /search

搜索相似文檔

**請求格式:**

```json
{
  "query": "搜索查詢",
  "k": 5
}
```

**回應格式:**

```json
{
  "success": true,
  "query": "搜索查詢",
  "results": [
    {
      "content": "文檔內容片段",
      "metadata": {
        "source": "來源",
        "doc_id": "文檔ID"
      },
      "score": 0.85
    }
  ]
}
```

### POST /chunking-strategy

獲取文檔分割策略建議

**請求格式:**

```json
{
  "text": "要分析的文本內容"
}
```

**回應格式:**

```json
{
  "success": true,
  "strategy": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "text_analysis": {
    "length": 5000,
    "lines": 50,
    "words": 800,
    "characters": 5000
  }
}
```

## 配置說明

### 環境變量

| 變量名                | 說明                   | 預設值             | 必需 |
| --------------------- | ---------------------- | ------------------ | ---- |
| `PINECONE_API_KEY`    | Pinecone API 金鑰      | -                  | ✓    |
| `GEMINI_API_KEY`      | Google Gemini API 金鑰 | -                  | ✓    |
| `PINECONE_ENV`        | Pinecone 環境          | us-east-1          | ✗    |
| `PINECONE_INDEX_NAME` | 索引名稱               | langchain-rag-demo | ✗    |

### 模型配置

- **嵌入模型**: sentence-transformers/all-roberta-large-v1 (1024 維)
- **LLM**: Google Gemini 2.5 Flash
- **向量相似度**: Cosine 相似度
- **分割器**: RecursiveCharacterTextSplitter
- **檢索器**: 相似性搜索 (k=3)

### 文檔分割策略

系統會根據文檔長度自動選擇分割策略：

| 文檔長度    | Chunk Size | Chunk Overlap | 適用情境 |
| ----------- | ---------- | ------------- | -------- |
| < 2,000 字  | 500        | 50            | 短文檔   |
| < 10,000 字 | 1,000      | 200           | 中等文檔 |
| < 50,000 字 | 1,500      | 300           | 長文檔   |
| ≥ 50,000 字 | 2,000      | 400           | 超長文檔 |

### 支援文件格式

- **PDF**: 使用 pdfplumber (優先) 和 PyPDF2 (備用)
- **TXT**: UTF-8 編碼文字文件
- **直接文字**: 透過 API 直接上傳文字內容

### Flask 配置

- **Host**: 0.0.0.0 (接受所有連接)
- **Port**: 5001
- **Debug**: True (開發模式)
- **最大文件大小**: 16MB
- **上傳文件夾**: uploads/

## 項目結構

```
LangChain/
├── app.py                  # 主要 Flask 應用程序
├── frontend.html           # Web 前端界面
├── requirements.txt        # Python 依賴列表
├── README.md              # 項目說明文檔
├── run.py                 # 啟動腳本
├── start.ps1              # PowerShell 啟動腳本
├── test_document.txt      # 測試文檔
├── .env                   # 環境變量配置 (需自行創建)
├── uploads/               # 上傳文件臨時存儲
├── langchainenv/          # Python 虛擬環境
└── __pycache__/           # Python 編譯緩存
```

## 核心組件說明

### LangChainRAGSystem 類

核心 RAG 系統實現，包含以下主要方法：

- `initialize_components()`: 初始化所有 LangChain 組件
- `add_documents()`: 添加文檔到向量存儲
- `process_file()`: 處理 PDF/TXT 文件
- `query()`: 執行問答查詢
- `search_documents()`: 搜索相似文檔
- `get_chunking_strategy_recommendation()`: 推薦分割策略

### 主要特色功能

1. **智能文檔分割**: 根據內容長度自動選擇最佳分割策略
2. **多格式支援**: PDF 和 TXT 文件自動處理
3. **錯誤恢復**: PDF 處理失敗時自動切換備用解析器
4. **自定義提示**: 針對繁體中文優化的問答提示模板
5. **來源追蹤**: 保留文檔來源和元數據信息
6. **即時統計**: 實時監控向量數據庫狀態

## 故障排除

### 1. 依賴安裝問題

```bash
# 升級 pip
pip install --upgrade pip

# 清除緩存重新安裝
pip cache purge
pip install -r requirements.txt --no-cache-dir

# Windows 用戶如果遇到編譯問題
pip install --only-binary=all -r requirements.txt
```

### 2. API 金鑰問題

- 確保 `.env` 文件存在且格式正確
- 檢查 API 金鑰是否有效且未過期
- 確認 API 配額是否充足
- Pinecone 金鑰格式：`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- Gemini 金鑰格式：`AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`

### 3. 向量數據庫連接問題

- 檢查 Pinecone API 金鑰和環境設定
- 確認網絡連接正常
- 查看 Pinecone 服務狀態：https://status.pinecone.io/
- 檢查 Pinecone 索引是否正確創建

### 4. 模型載入問題

- 確保有足夠的磁盤空間 (>2GB)
- 檢查網絡連接（首次下載模型需要穩定網絡）
- 如果下載緩慢，可考慮使用 Hugging Face 鏡像

### 5. PDF 處理問題

- 確保 PDF 文件沒有密碼保護
- 檢查 PDF 文件是否包含可提取的文字 (非純圖片)
- 對於複雜 PDF，系統會自動切換解析器

### 6. 記憶體問題

- 大型文檔可能需要較多記憶體
- 考慮調整 chunk_size 參數減少記憶體使用
- 在低記憶體環境下，可以一次處理較少文檔

### 7. 端口佔用問題

```bash
# Windows 查看端口佔用
netstat -ano | findstr :5001

# 殺死佔用進程 (替換 PID)
taskkill /F /PID <PID>
```

### 8. 虛擬環境問題

```bash
# 重新創建虛擬環境
deactivate
rmdir /s langchainenv
python -m venv langchainenv
langchainenv\Scripts\activate
pip install -r requirements.txt
```

## 效能優化建議

### 1. 文檔分割優化

- 短文檔 (<2000 字): 使用較小的 chunk_size (500)
- 長文檔 (>10000 字): 使用較大的 chunk_size (1500+)
- 技術文檔: 增加 chunk_overlap 確保上下文連貫性

### 2. 檢索優化

- 調整檢索數量 (k) 根據需求平衡精度和速度
- 對於特定領域，可以考慮使用領域特定的嵌入模型

### 3. 系統資源

- 生產環境建議使用 GPU 加速嵌入計算
- 考慮使用快取機制減少重複計算
- 對於高併發場景，考慮使用負載平衡

## 開發指南

### 添加新功能

1. 在 `LangChainRAGSystem` 類中添加新方法
2. 在 Flask 路由中創建對應 API 端點
3. 在前端 `frontend.html` 中添加相應的界面和邏輯
4. 更新 API 文檔和使用說明

### 自定義嵌入模型

修改 `LangChainRAGSystem.initialize_components()` 中的嵌入模型：

```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="your-preferred-model",  # 例如: "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs={'device': 'cpu'}
)
```

### 自定義提示模板

在 `initialize_components()` 方法中修改 `custom_prompt` 變量來自定義 AI 回答風格。

### 添加新的文件格式支援

在 `process_file()` 方法中添加新的文件處理邏輯：

```python
elif file_extension == '.docx':
    # 添加 DOCX 處理邏輯
    pass
```

## 版本歷史

### v1.0.0 (當前版本)

- ✅ 完整的 LangChain RAG 系統實現
- ✅ 支援 PDF 和 TXT 文件上傳
- ✅ 智能文檔分割策略
- ✅ Google Gemini 2.5 Flash 集成
- ✅ Pinecone 向量數據庫支援
- ✅ 現代化 Web 前端界面
- ✅ 完整的 API 文檔和錯誤處理

### 規劃中的功能

- 🔄 支援更多文件格式 (DOCX, XLSX)
- 🔄 批量文檔上傳
- 🔄 文檔管理和刪除功能
- 🔄 對話歷史記錄
- 🔄 多語言支援
- 🔄 用戶身份驗證
- 🔄 API 限流和快取

## 貢獻指南

歡迎貢獻代碼！請遵循以下步驟：

1. Fork 本項目
2. 創建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打開 Pull Request

### 開發規範

- 代碼註釋使用繁體中文
- 遵循 PEP 8 Python 編碼規範
- 添加適當的錯誤處理和日誌記錄
- 更新相應的文檔和測試

## 許可證

本項目採用 MIT 許可證 - 查看 [LICENSE](LICENSE) 文件了解詳情。

## 聯繫方式

如有問題或建議，請：

- 創建 GitHub Issue
- 發送郵件至維護者
- 參與項目討論

## 致謝

感謝以下開源項目：

- [LangChain](https://github.com/langchain-ai/langchain) - 核心 RAG 框架
- [Pinecone](https://www.pinecone.io/) - 向量數據庫服務
- [Google Generative AI](https://ai.google.dev/) - 大語言模型
- [Sentence Transformers](https://www.sbert.net/) - 文本嵌入模型
- [Flask](https://flask.palletsprojects.com/) - Web 框架

---

**⚠️ 重要提示**:

- 請妥善保管您的 API 金鑰，不要將其提交到版本控制系統中
- 生產環境使用前請充分測試所有功能
- 建議定期備份重要的向量數據

**📖 更多資源**:

- [LangChain 官方文檔](https://python.langchain.com/)
- [Pinecone 文檔](https://docs.pinecone.io/)
- [Google AI 文檔](https://ai.google.dev/docs)
