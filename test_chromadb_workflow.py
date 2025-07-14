import asyncio
import os
from src.notebookllama.utils import process_file, query_index

SAMPLE_DOC = "sample_test_doc.txt"
SAMPLE_TEXT = """
NotebookLlama is an open-source alternative to NotebookLM. It uses Ollama for local LLM inference and ChromaDB for vector storage. The project is maintained by mrbanks2321.
"""

async def main():
    print("\n--- ChromaDB Workflow Test ---\n")
    # 1. Write a sample document
    with open(SAMPLE_DOC, "w") as f:
        f.write(SAMPLE_TEXT)
    print(f"Sample document '{SAMPLE_DOC}' created.")

    # 2. Add the document to the index
    print("Adding document to ChromaDB via process_file...")
    print("Before await process_file")
    try:
        result, _ = await process_file(SAMPLE_DOC)
        print("After await process_file")
        if result is None:
            print("process_file returned None.\n")
        else:
            print(f"process_file result: {result[:100]}...\n")
    except Exception as e:
        print(f"Exception during process_file: {e}")

    # 3. Query for a fact in the document
    question = "Who maintains NotebookLlama?"
    print(f"Querying: {question}")
    answer = await query_index(question)
    print(f"Answer:\n{answer}\n")

    # 4. Edge case: Query for something not in the doc
    question2 = "What is the capital of France?"
    print(f"Querying: {question2}")
    answer2 = await query_index(question2)
    print(f"Answer:\n{answer2}\n")

    # 5. Edge case: Add an empty document
    with open("empty_doc.txt", "w") as f:
        f.write("")
    print("Adding empty document...")
    result_empty, _ = await process_file("empty_doc.txt")
    print(f"process_file (empty) result: {result_empty}\n")

    # 6. Edge case: Add the same document again (duplicate)
    print("Adding duplicate document...")
    result_dup, _ = await process_file(SAMPLE_DOC)
    print(f"process_file (duplicate) result: {result_dup[:100]}...\n")

    # Cleanup
    os.remove(SAMPLE_DOC)
    os.remove("empty_doc.txt")
    print("\nTest complete. Sample files removed.")

if __name__ == "__main__":
    asyncio.run(main()) 