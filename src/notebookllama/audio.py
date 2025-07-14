import tempfile as temp
import os
import uuid
from dotenv import load_dotenv

from pydub import AudioSegment
from elevenlabs import AsyncElevenLabs
from llama_index.core.llms.structured_llm import StructuredLLM
from typing_extensions import Self
from typing import List, Literal
from pydantic import BaseModel, ConfigDict, model_validator, Field
from llama_index.core.llms import ChatMessage
from src.notebookllama.ollama_llm import OllamaLLM
from TTS.api import TTS as CoquiTTS


class ConversationTurn(BaseModel):
    speaker: Literal["speaker1", "speaker2"] = Field(
        description="The person who is speaking",
    )
    content: str = Field(
        description="The content of the speech",
    )


class MultiTurnConversation(BaseModel):
    conversation: List[ConversationTurn] = Field(
        description="List of conversation turns. Conversation must start with speaker1, and continue with an alternance of speaker1 and speaker2",
        min_length=3,
        max_length=50,
        examples=[
            [
                ConversationTurn(speaker="speaker1", content="Hello, who are you?"),
                ConversationTurn(
                    speaker="speaker2", content="I am very well, how about you?"
                ),
                ConversationTurn(speaker="speaker1", content="I am well too, thanks!"),
            ]
        ],
    )

    @model_validator(mode="after")
    def validate_conversation(self) -> Self:
        speakers = [turn.speaker for turn in self.conversation]
        if speakers[0] != "speaker1":
            raise ValueError("Conversation must start with speaker1")
        for i, speaker in enumerate(speakers):
            if i % 2 == 0 and speaker != "speaker1":
                raise ValueError(
                    "Conversation must be an alternance between speaker1 and speaker2"
                )
            elif i % 2 != 0 and speaker != "speaker2":
                raise ValueError(
                    "Conversation must be an alternance between speaker1 and speaker2"
                )
            continue
        return self


class PodcastGenerator(BaseModel):
    llm: StructuredLLM
    client: object  # Accepts either AsyncElevenLabs or CoquiTTS
    coqui_tts: object = None  # Optional CoquiTTS fallback

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_podcast(self) -> Self:
        try:
            assert self.llm.output_cls == MultiTurnConversation
        except AssertionError:
            raise ValueError(
                f"The output class of the structured LLM must be {MultiTurnConversation.__qualname__}, your LLM has output class: {self.llm.output_cls.__qualname__}"
            )
        return self

    async def _conversation_script(self, file_transcript: str) -> MultiTurnConversation:
        response = await self.llm.achat(
            messages=[
                ChatMessage(
                    role="user",
                    content=f"Please create a multi-turn conversation with two speakers starting from this file transcript:\n\n'''\n{file_transcript}\n'''",
                )
            ]
        )
        return MultiTurnConversation.model_validate_json(response.message.content)

    async def _generate_elevenlabs_audio(self, text, voice_id, output_format, model_id):
        speech_iterator = self.client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            output_format=output_format,
            model_id=model_id,
        )
        audio_bytes = b""
        async for chunk in speech_iterator:
            if chunk:
                audio_bytes += chunk
        return audio_bytes

    def _generate_coqui_audio(self, text, speaker="en-US-001", output_path=None):
        # Coqui TTS is synchronous
        if not self.coqui_tts:
            self.coqui_tts = CoquiTTS(model_name="tts_models/en/vctk/vits")
        if output_path is None:
            output_path = f"coqui_{str(uuid.uuid4())}.wav"
        self.coqui_tts.tts_to_file(text=text, file_path=output_path)
        return output_path

    async def _conversation_audio(self, conversation: MultiTurnConversation) -> str:
        files: List[str] = []
        for turn in conversation.conversation:
            temp_path = None
            try:
                # Try ElevenLabs first
                audio_bytes = await self._generate_elevenlabs_audio(
                    text=turn.content,
                    voice_id="nPczCjzI2devNBz1zQrb" if turn.speaker == "speaker1" else "Xb7hH8MSUJpSbSDYk0k2",
                    output_format="mp3_22050_32",
                    model_id="eleven_turbo_v2_5",
                )
                fl = temp.NamedTemporaryFile(suffix=".mp3", delete=False)
                with open(fl.name, "wb") as f:
                    f.write(audio_bytes)
                fl.close()
                temp_path = fl.name
            except Exception as e:
                # Fallback to Coqui TTS
                print(f"[WARN] ElevenLabs failed: {e}. Falling back to Coqui TTS.")
                temp_path = self._generate_coqui_audio(turn.content)
            files.append(temp_path)

        output_path = f"conversation_{str(uuid.uuid4())}.mp3"
        combined_audio: AudioSegment = AudioSegment.empty()

        for file_path in files:
            audio = AudioSegment.from_file(file_path)
            combined_audio += audio
            os.remove(file_path)

        # Export with high quality MP3 settings
        combined_audio.export(
            output_path,
            format="mp3",
            bitrate="320k",  # High quality bitrate
            parameters=["-q:a", "0"],  # Highest quality
        )
        return output_path

    async def create_conversation(self, file_transcript: str):
        conversation = await self._conversation_script(file_transcript=file_transcript)
        podcast_file = await self._conversation_audio(conversation=conversation)
        return podcast_file


load_dotenv()

# Remove OpenAI API key check and SLLM instantiation
# if os.getenv("ELEVENLABS_API_KEY", None) and os.getenv("OPENAI_API_KEY", None):
#     SLLM = OpenAIResponses(
#         model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY")
#     ).as_structured_llm(MultiTurnConversation)
# else:
#     SLLM = None
# Replace with Ollama LLM instance
ollama_llm = OllamaLLM(model="gemma3:4b", temperature=0.1)
SLLM = ollama_llm.as_structured_llm(MultiTurnConversation)
EL_CLIENT = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
# Add CoquiTTS fallback client
COQUI_TTS = None  # Will be instantiated on demand
PODCAST_GEN = PodcastGenerator(llm=SLLM, client=EL_CLIENT, coqui_tts=COQUI_TTS)
