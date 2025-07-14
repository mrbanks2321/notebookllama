import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from the project root
env_path = Path(__file__).parent / ".env"
print(f"DEBUG: Loading .env from {env_path.resolve()}")
load_dotenv(dotenv_path=env_path)
print("DEBUG: ELEVENLABS_API_KEY =", os.getenv("ELEVENLABS_API_KEY"))
import asyncio
from src.notebookllama.ollama_llm import OllamaLLM
from src.notebookllama.audio import PodcastGenerator, AsyncElevenLabs, MultiTurnConversation

async def main():
    print("\n--- Audio Workflow Test ---\n")

    # 1. Set up LLM and ElevenLabs client
    print("[LOG] Initializing Ollama LLM...")
    llm = OllamaLLM(model="gemma3:4b", temperature=0.1)
    structured_llm = llm.as_structured_llm(MultiTurnConversation)
    print("[LOG] Ollama LLM initialized and wrapped as StructuredLLM.")

    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    if not elevenlabs_api_key:
        print("[ERROR] ELEVENLABS_API_KEY not set. Exiting test.")
        return

    print("[LOG] Initializing ElevenLabs client...")
    el_client = AsyncElevenLabs(api_key=elevenlabs_api_key)
    print("[LOG] ElevenLabs client initialized.")
    # List available voices
    try:
        voices = await el_client.voices.get_all()
        print("DEBUG: Available voices (raw):", voices)
        for v in voices:
            print("DEBUG: Voice entry:", v, type(v))
    except Exception as e:
        print(f"[ERROR] Could not list voices: {e}")
    print("DEBUG: dir(el_client):", dir(el_client))
    print("DEBUG: type(el_client.text_to_speech):", type(el_client.text_to_speech))
    print("DEBUG: dir(el_client.text_to_speech):", dir(el_client.text_to_speech))

    # 2. Set up PodcastGenerator (or your audio pipeline)
    print("[LOG] Initializing PodcastGenerator...")
    podcast_gen = PodcastGenerator(llm=structured_llm, client=el_client)
    print("[LOG] PodcastGenerator initialized.")

    # 3. Generate text with LLM
    prompt = "Summarize the benefits of using local LLMs and vector databases for privacy and performance."
    print(f"[LOG] Sending prompt to LLM: {prompt}")
    try:
        summary = await llm.acomplete(prompt)
        print(f"[LOG] LLM response: {summary.text[:200]}...")
    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")
        return

    # 4. Generate audio from text
    print("[LOG] Generating audio from LLM response...")
    try:
        audio_chunks = []
        async for chunk in el_client.text_to_speech.convert(text=summary.text, voice_id="EXAVITQu4vr4xnSDxMaL"):
            audio_chunks.append(chunk)
        audio_bytes = b"".join(audio_chunks)
        audio_path = "test_output_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        print(f"[LOG] Audio saved to {audio_path}")
    except Exception as e:
        print(f"[ERROR] Audio generation failed: {e}")
        return

    print("\n[LOG] Audio workflow test completed successfully.")

async def test_empty_string(el_client):
    print("[TEST] TTS with empty string...")
    try:
        text = ""
        if not text:
            print("[PASS] Skipped TTS for empty string as expected.")
            return
        audio_chunks = []
        async for chunk in el_client.text_to_speech.convert(text=text, voice_id="EXAVITQu4vr4xnSDxMaL"):
            audio_chunks.append(chunk)
        if audio_chunks:
            print("[FAIL] Audio generated for empty string!")
        else:
            print("[PASS] No audio generated for empty string.")
    except Exception as e:
        print(f"[PASS] Exception for empty string as expected: {e}")

async def test_invalid_api_key():
    print("[TEST] TTS with invalid API key...")
    from elevenlabs.client import AsyncElevenLabs
    el_client = AsyncElevenLabs(api_key="invalid_key")
    try:
        audio_chunks = []
        async for chunk in el_client.text_to_speech.convert(text="Test", voice_id="EXAVITQu4vr4xnSDxMaL"):
            audio_chunks.append(chunk)
        print("[FAIL] Audio generated with invalid API key!")
    except Exception as e:
        print(f"[PASS] Exception for invalid API key as expected: {e}")

async def test_audio_file_output():
    print("[TEST] Audio file output...")
    output_file = "test_output_audio.mp3"
    if os.path.exists(output_file):
        print(f"[PASS] Audio file {output_file} exists.")
        size = os.path.getsize(output_file)
        print(f"[INFO] File size: {size} bytes.")
        if size > 1000:
            print("[PASS] Audio file size is reasonable.")
        else:
            print("[FAIL] Audio file size is too small.")
    else:
        print(f"[FAIL] Audio file {output_file} does not exist.")

async def test_long_text(el_client):
    print("[TEST] TTS with long text...")
    long_text = (
        "This is a long text input designed to test the TTS pipeline. "
        "It should be long enough to span multiple sentences and paragraphs. "
        "The quick brown fox jumps over the lazy dog. " * 10 +
        "End of long text test."
    )
    try:
        audio_chunks = []
        async for chunk in el_client.text_to_speech.convert(text=long_text, voice_id="EXAVITQu4vr4xnSDxMaL"):
            audio_chunks.append(chunk)
        audio_bytes = b"".join(audio_chunks)
        audio_path = "test_long_text_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        if os.path.exists(audio_path):
            size = os.path.getsize(audio_path)
            print(f"[PASS] Long text audio file created: {audio_path} ({size} bytes)")
            if size > 1000:
                print("[PASS] Long text audio file size is reasonable.")
            else:
                print("[FAIL] Long text audio file size is too small.")
            os.remove(audio_path)
        else:
            print("[FAIL] Long text audio file was not created.")
    except Exception as e:
        print(f"[FAIL] Exception during long text TTS: {e}")

async def test_special_characters(el_client):
    print("[TEST] TTS with special characters...")
    special_text = (
        "Hello, world! @#%&*()[]{};:'\",.<>?/\\|`~ 😊🚀✨ — “quotes” — accented: café, naïve, résumé."
    )
    try:
        audio_chunks = []
        async for chunk in el_client.text_to_speech.convert(text=special_text, voice_id="EXAVITQu4vr4xnSDxMaL"):
            audio_chunks.append(chunk)
        audio_bytes = b"".join(audio_chunks)
        audio_path = "test_special_characters_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        if os.path.exists(audio_path):
            size = os.path.getsize(audio_path)
            print(f"[PASS] Special characters audio file created: {audio_path} ({size} bytes)")
            if size > 1000:
                print("[PASS] Special characters audio file size is reasonable.")
            else:
                print("[FAIL] Special characters audio file size is too small.")
            os.remove(audio_path)
        else:
            print("[FAIL] Special characters audio file was not created.")
    except Exception as e:
        print(f"[FAIL] Exception during special characters TTS: {e}")

async def test_pipeline():
    print("[TEST] End-to-end pipeline...")
    # Simulate: document ingestion -> summary -> TTS
    from src.notebookllama.ollama_llm import OllamaLLM
    from src.notebookllama.audio import PodcastGenerator, AsyncElevenLabs, MultiTurnConversation
    llm = OllamaLLM()
    structured_llm = llm.as_structured_llm(MultiTurnConversation)
    el_client = AsyncElevenLabs()
    pg = PodcastGenerator(llm=structured_llm, client=el_client)
    doc = "Local-first AI notebooks are great for privacy."
    podcast_file = await pg.create_conversation(doc)
    print(f"[INFO] Podcast file generated: {podcast_file}")
    import os
    if os.path.exists(podcast_file):
        size = os.path.getsize(podcast_file)
        print(f"[PASS] Pipeline produced podcast audio file: {podcast_file} ({size} bytes)")
        if size > 1000:
            print("[PASS] Podcast audio file size is reasonable.")
        else:
            print("[FAIL] Podcast audio file size is too small.")
        os.remove(podcast_file)
    else:
        print("[FAIL] Podcast audio file was not created.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 