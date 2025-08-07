import os
import logging

# 在導入 pinecone 之前設置環境變數以禁用插件檢查
os.environ['PINECONE_SKIP_PLUGIN_CHECK'] = 'true'

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback
from typing import List, Dict, Any, Optional
import PyPDF2
import pdfplumber
from werkzeug.utils import secure_filename

# LangChain 導入
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Pinecone 導入
from pinecone import Pinecone, ServerlessSpec

# 環境變數
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 設置上傳文件夾
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 最大文件大小

# 配置
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-rag-demo")

class LangChainRAGSystem:
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.pinecone_client = None
        self.initialize_components()
    
    def initialize_components(self):
        """初始化 LangChain 組件"""
        try:
            # 檢查必要的環境變數
            if not PINECONE_API_KEY or not GEMINI_API_KEY:
                raise ValueError("請設置 PINECONE_API_KEY 和 GEMINI_API_KEY 環境變數")
            
            # 初始化嵌入模型 (使用 1024 維度)
            logger.info("初始化 HuggingFace Embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-roberta-large-v1",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("嵌入模型初始化成功")
            
            # 初始化 LLM
            logger.info("初始化 Google Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.1
            )
            logger.info("LLM 初始化成功")
            
            # 初始化 Pinecone
            logger.info("初始化 Pinecone...")
            self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            
            # 檢查索引是否存在
            existing_indexes = [index_info.name for index_info in self.pinecone_client.list_indexes()]
            
            if INDEX_NAME not in existing_indexes:
                logger.info(f"創建 Pinecone 索引: {INDEX_NAME}")
                self.pinecone_client.create_index(
                    name=INDEX_NAME,
                    dimension=1024,  # 使用 1024 維度
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=PINECONE_ENV
                    )
                )
                # 等待索引創建完成
                import time
                time.sleep(10)
            
            # 初始化向量存儲
            logger.info("初始化向量存儲...")
            self.vectorstore = PineconeVectorStore(
                index_name=INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=PINECONE_API_KEY
            )
            logger.info("向量存儲初始化成功")
            
            # 創建自定義提示詞模板
            custom_prompt = PromptTemplate(
                template="""你是一個專業的AI助手。請根據以下提供的上下文資訊來回答用戶的問題。

                上下文資訊:
                {context}

                用戶問題: {question}

                請根據上述上下文資訊回答問題。如果上下文中沒有足夠的資訊來回答問題，請誠實地說明，並基於你的一般知識提供有幫助的回答。

                回答指引：
                1. 優先使用上下文中的資訊來回答問題
                2. 如果上下文包含相關資訊，請直接引用或概括相關內容
                3. 如果上下文資訊不足或不相關，請明確說明並提供一般性的幫助
                4. 使用繁體中文回答
                5. 結構清晰、邏輯分明

                回答:""",
                input_variables=["context", "question"]
            )
            
            # 初始化 RetrievalQA 鏈
            logger.info("初始化檢索問答鏈...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                chain_type_kwargs={"prompt": custom_prompt},
                return_source_documents=True
            )
            logger.info("檢索問答鏈初始化成功")
            
        except Exception as e:
            logger.error(f"初始化組件時發生錯誤: {str(e)}")
            raise
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
        """添加文檔到向量存儲"""
        try:
            # 創建文檔對象
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                documents.append(Document(page_content=text, metadata=metadata))
            
            # 智能分割策略
            if chunk_size is None or chunk_overlap is None:
                # 根據文檔長度自動決定分割策略
                total_length = sum(len(doc.page_content) for doc in documents)
                if total_length < 2000:
                    # 短文檔：不分割或小塊分割
                    chunk_size = chunk_size or 500
                    chunk_overlap = chunk_overlap or 50
                elif total_length < 10000:
                    # 中等文檔：標準分割
                    chunk_size = chunk_size or 1000
                    chunk_overlap = chunk_overlap or 200
                else:
                    # 長文檔：大塊分割
                    chunk_size = chunk_size or 1500
                    chunk_overlap = chunk_overlap or 300
                    
                logger.info(f"自動選擇分割策略：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            else:
                logger.info(f"使用自定義分割策略：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            
            # 文字分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
            )
            split_docs = text_splitter.split_documents(documents)
            
            # 添加到向量存儲
            doc_ids = self.vectorstore.add_documents(split_docs)
            logger.info(f"成功添加 {len(split_docs)} 個文檔塊到索引 {INDEX_NAME}")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"添加文檔時發生錯誤: {str(e)}")
            raise
    
    def process_file(self, file_path: str) -> str:
        """處理文件並提取文字內容"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.txt':
                # 處理 TXT 文件
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                logger.info(f"成功讀取 TXT 文件: {file_path}")
                return content
                
            elif file_extension == '.pdf':
                # 處理 PDF 文件
                content = self._extract_pdf_content(file_path)
                logger.info(f"成功讀取 PDF 文件: {file_path}")
                return content
                
            else:
                raise ValueError(f"不支援的文件格式: {file_extension}")
                
        except Exception as e:
            logger.error(f"處理文件時發生錯誤: {str(e)}")
            raise
    
    def _extract_pdf_content(self, file_path: str) -> str:
        """從 PDF 文件提取文字內容"""
        content = ""
        
        try:
            # 嘗試使用 pdfplumber（更好的文字提取）
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        content += f"\n--- 第 {page_num + 1} 頁 ---\n"
                        content += page_text + "\n"
            
            if content.strip():
                return content
        except Exception as e:
            logger.warning(f"pdfplumber 提取失敗: {e}，嘗試使用 PyPDF2")
        
        try:
            # 備用方案：使用 PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        content += f"\n--- 第 {page_num + 1} 頁 ---\n"
                        content += page_text + "\n"
            
            return content
            
        except Exception as e:
            raise ValueError(f"無法提取 PDF 內容: {e}")
    
    def get_chunking_strategy_recommendation(self, text: str) -> Dict[str, int]:
        """根據文本特徵推薦分割策略"""
        length = len(text)
        lines = text.count('\n')
        
        if length < 2000:
            return {"chunk_size": 500, "chunk_overlap": 50}
        elif length < 10000:
            return {"chunk_size": 1000, "chunk_overlap": 200}
        elif length < 50000:
            return {"chunk_size": 1500, "chunk_overlap": 300}
        else:
            return {"chunk_size": 2000, "chunk_overlap": 400}
    
    def query(self, question: str) -> Dict[str, Any]:
        """處理查詢"""
        try:
            logger.info(f"處理查詢: {question}")
            
            # 執行查詢
            result = self.qa_chain.invoke({"query": question})
            
            # 格式化回應
            response = {
                'success': True,
                'query': question,
                'answer': result['result'],
                'source_documents': []
            }
            
            # 添加來源文檔資訊
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    response['source_documents'].append({
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    })
            
            logger.info("查詢處理成功")
            return response
            
        except Exception as e:
            logger.error(f"處理查詢時發生錯誤: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似文檔"""
        try:
            # 執行相似性搜索
            docs = self.vectorstore.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"搜索文檔時發生錯誤: {str(e)}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """獲取索引統計資訊"""
        try:
            index = self.pinecone_client.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            return {
                'total_vector_count': stats.total_vector_count,
                'dimension': stats.dimension
            }
        except Exception as e:
            logger.error(f"獲取索引統計時發生錯誤: {str(e)}")
            return {}

# 輔助函數
def allowed_file(filename):
    """檢查文件擴展名是否被允許"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 初始化 RAG 系統
try:
    rag_system = LangChainRAGSystem()
    logger.info("RAG 系統初始化成功")
except Exception as e:
    logger.error(f"RAG 系統初始化失敗: {str(e)}")
    rag_system = None

# API 路由
@app.route('/health', methods=['GET'])
def health_check():
    """健康檢查"""
    try:
        if not rag_system:
            return jsonify({
                'status': 'unhealthy',
                'error': 'RAG 系統未初始化'
            }), 500
        
        stats = rag_system.get_index_stats()
        
        return jsonify({
            'status': 'healthy',
            'components': {
                'embedding_model': 'sentence-transformers/all-roberta-large-v1',
                'vector_store': 'Pinecone',
                'llm': 'Gemini-2.5-Flash',
                'qa_chain': 'RetrievalQA'
            },
            'pinecone_stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/query', methods=['POST'])
def handle_query():
    """處理查詢請求"""
    try:
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG 系統未初始化'
            }), 500
        
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': '查詢不能為空'
            }), 400
        
        # 處理查詢
        result = rag_system.query(query)
        
        if result['success']:
            # 格式化回應以匹配前端期望的格式
            context = '\n\n'.join([doc['content'] for doc in result['source_documents']])
            
            return jsonify({
                'success': True,
                'response': result['answer'],
                'context': context if context else '無相關上下文資訊',
                'source_documents_count': len(result['source_documents'])
            })
        else:
            return jsonify(result), 500
        
    except Exception as e:
        logger.error(f"處理查詢請求時發生錯誤: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/upload', methods=['POST'])
def upload_document():
    """上傳文檔（支援文字和文件）"""
    try:
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG 系統未初始化'
            }), 500
        
        # 檢查是否是文件上傳
        if 'file' in request.files:
            return handle_file_upload()
        else:
            return handle_text_upload()
            
    except Exception as e:
        logger.error(f"上傳文檔時發生錯誤: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def handle_file_upload():
    """處理文件上傳"""
    file = request.files['file']
    doc_id = request.form.get('doc_id', '')
    source = request.form.get('source', '文件上傳')
    chunk_size = request.form.get('chunk_size', type=int)
    chunk_overlap = request.form.get('chunk_overlap', type=int)
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': '沒有選擇文件'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': '不支援的文件格式，請上傳 TXT 或 PDF 文件'
        }), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not doc_id:
            doc_id = os.path.splitext(filename)[0]
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # 提取文件內容
            content = rag_system.process_file(file_path)
            
            # 如果沒有指定分割參數，提供推薦策略
            if chunk_size is None and chunk_overlap is None:
                strategy = rag_system.get_chunking_strategy_recommendation(content)
                chunk_size = strategy['chunk_size']
                chunk_overlap = strategy['chunk_overlap']
            
            # 添加文檔
            metadata = {
                'source': source,
                'doc_id': doc_id,
                'filename': filename,
                'file_type': os.path.splitext(filename)[1],
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'content_length': len(content)
            }
            
            doc_ids = rag_system.add_documents(
                [content], 
                [metadata], 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            # 清理臨時文件
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'message': f'文件 "{filename}" 已成功上傳到索引 "{INDEX_NAME}"，生成 {len(doc_ids)} 個文檔塊',
                'doc_id': doc_id,
                'doc_ids': doc_ids,
                'content_length': len(content),
                'chunk_count': len(doc_ids),
                'chunking_strategy': {
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap
                }
            })
            
        except Exception as e:
            # 清理臨時文件
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

def handle_text_upload():
    """處理文字上傳"""
    data = request.get_json()
    doc_id = data.get('id', '')
    text = data.get('text', '')
    source = data.get('source', '文字上傳')
    chunk_size = data.get('chunk_size', type=int)
    chunk_overlap = data.get('chunk_overlap', type=int)
    
    if not text:
        return jsonify({
            'success': False,
            'error': '文字內容不能為空'
        }), 400
    
    # 如果沒有指定分割參數，提供推薦策略
    if chunk_size is None and chunk_overlap is None:
        strategy = rag_system.get_chunking_strategy_recommendation(text)
        chunk_size = strategy['chunk_size']
        chunk_overlap = strategy['chunk_overlap']
    
    # 添加文檔
    metadata = {
        'source': source,
        'doc_id': doc_id,
        'content_type': 'text',
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'content_length': len(text)
    }
    
    doc_ids = rag_system.add_documents(
        [text], 
        [metadata], 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    return jsonify({
        'success': True,
        'message': f'文字內容已成功上傳到索引 "{INDEX_NAME}"，生成 {len(doc_ids)} 個文檔塊',
        'doc_id': doc_id,
        'doc_ids': doc_ids,
        'content_length': len(text),
        'chunk_count': len(doc_ids),
        'chunking_strategy': {
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
    })

@app.route('/chunking-strategy', methods=['POST'])
def get_chunking_strategy():
    """根據文本內容推薦分割策略"""
    try:
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG 系統未初始化'
            }), 500
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'success': False,
                'error': '文字內容不能為空'
            }), 400
        
        strategy = rag_system.get_chunking_strategy_recommendation(text)
        
        return jsonify({
            'success': True,
            'strategy': strategy,
            'text_analysis': {
                'length': len(text),
                'lines': text.count('\n'),
                'words': len(text.split()),
                'characters': len(text)
            }
        })
        
    except Exception as e:
        logger.error(f"獲取分割策略時發生錯誤: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/search', methods=['POST'])
def search_documents():
    """搜索文檔"""
    try:
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG 系統未初始化'
            }), 500
        
        data = request.get_json()
        query = data.get('query', '').strip()
        k = data.get('k', 5)
        
        if not query:
            return jsonify({
                'success': False,
                'error': '搜索查詢不能為空'
            }), 400
        
        # 搜索文檔
        results = rag_system.search_documents(query, k)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"搜索文檔時發生錯誤: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 靜態文件路由
@app.route('/')
def index():
    """提供前端 HTML 文件"""
    return send_from_directory('.', 'frontend.html')

@app.route('/frontend.html')
def frontend():
    """提供前端 HTML 文件"""
    return send_from_directory('.', 'frontend.html')

if __name__ == '__main__':
    if rag_system:
        logger.info("啟動 LangChain RAG Flask 應用程序...")
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        logger.error("無法啟動應用程序 - RAG 系統初始化失敗")