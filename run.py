#!/usr/bin/env python3
"""
LangChain RAG 系統啟動腳本
"""

import sys
import subprocess
import os
from pathlib import Path

def check_requirements():
    """檢查必要的依賴是否已安裝"""
    try:
        import flask
        import langchain
        import pinecone
        import sentence_transformers
        import google.generativeai
        print("✅ 所有必要依賴已安裝")
        return True
    except ImportError as e:
        print(f"❌ 缺少依賴: {e}")
        print("請運行: pip install -r requirements.txt")
        return False

def check_env_file():
    """檢查環境配置文件"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  未找到 .env 文件")
        print("請複製 .env.example 為 .env 並填入您的 API 金鑰")
        return False
    print("✅ 找到環境配置文件")
    return True

def main():
    """主函數"""
    print("🚀 啟動 LangChain RAG 系統...")
    print("=" * 50)
    
    # 檢查依賴
    if not check_requirements():
        sys.exit(1)
    
    # 檢查環境文件
    if not check_env_file():
        print("繼續使用預設配置...")
    
    print("=" * 50)
    print("📊 系統信息:")
    print(f"Python 版本: {sys.version}")
    print(f"工作目錄: {os.getcwd()}")
    print("=" * 50)
    
    try:
        # 啟動 Flask 應用
        print("🌐 啟動 Web 服務器...")
        print("前端地址: http://localhost:8080 (請用瀏覽器打開 frontend.html)")
        print("API 地址: http://localhost:5001")
        print("按 Ctrl+C 停止服務器")
        print("=" * 50)
        
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except KeyboardInterrupt:
        print("\n👋 服務器已停止")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
