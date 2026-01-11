import pandas as pd
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config.settings import settings

def inspect_output():
    file_path = settings.OUTPUT_PATH
    
    if not os.path.exists(file_path):
        print(f"❌ No output file found at {file_path}. Run 'make run' first.")
        return

    print(f"🔍 Inspecting Pipeline Output: {file_path}")
    print("-" * 60)
    
    try:
        df = pd.read_parquet(file_path)
        
        # 1. High Level Stats
        total_chunks = len(df)
        unique_docs = df['parent_id'].nunique()
        
        print(f"✅ SUCCESSFULLY LOADED PARQUET FILE")
        print(f"📊 STATISTICS:")
        print(f"   • Total Processed Chunks:  {total_chunks}")
        print(f"   • Unique Input Documents:  {unique_docs}")
        if not df.empty:
            print(f"   • Vector Dimension:        {len(df.iloc[0]['vector'])}")
            print(f"   • Output Schema:           {list(df.columns)}")
        
        print("-" * 60)
        
        # 2. Sample Data
        if not df.empty:
            sample = df.iloc[0]
            print("📝 SAMPLE CHUNK (First Record):")
            print(f"   • Chunk ID:   {sample['chunk_id']}")
            print(f"   • Content Preview: \"{sample['text'][:100]}...\"")
            print(f"   • Vector Preview:  {sample['vector'][:5]}... (truncated)")
            print(f"   • Metadata:   {sample['metadata']}")
        
        print("-" * 60)
        print("💡 BUSINESS VALUE:")
        print("   This 'Silver Layer' data is now structured and ready for inputs into:")
        print("   1. Vector Databases (Pinecone/Weaviate) -> For Semantic Search")
        print("   2. Dashboarding Tools (Tableau/PowerBI) -> For Content Analytics")
        print("   3. LLM Context Windows -> For RAG Applications")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    inspect_output()
