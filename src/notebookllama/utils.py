from dotenv import load_dotenv
import pandas as pd
import json
import os
import uuid
import warnings
from datetime import datetime

from mrkdwn_analysis import MarkdownAnalyzer
from mrkdwn_analysis.markdown_analyzer import InlineParser, MarkdownParser
from pydantic import BaseModel, Field, model_validator
from llama_index.core.llms import ChatMessage, LLM
from llama_cloud_services import LlamaExtract, LlamaParse
from llama_cloud_services.extract import SourceText
from src.notebookllama.chroma_storage import ChromaStorage
from src.notebookllama.ollama_embeddings import OllamaEmbedding
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.base.response.schema import Response
from typing_extensions import override
from typing import List, Tuple, Union, Optional, Dict, cast
from typing_extensions import Self
from pyvis.network import Network
from src.notebookllama.ollama_llm import OllamaLLM

load_dotenv()


class MarkdownTextAnalyzer(MarkdownAnalyzer):
    @override
    def __init__(self, text: str):
        self.text = text
        parser = MarkdownParser(self.text)
        self.tokens = parser.parse()
        self.references = parser.references
        self.footnotes = parser.footnotes
        self.inline_parser = InlineParser(
            references=self.references, footnotes=self.footnotes
        )
        self._parse_inline_tokens()


class Node(BaseModel):
    id: str
    content: str


class Edge(BaseModel):
    from_id: str
    to_id: str


class MindMap(BaseModel):
    nodes: List[Node] = Field(
        description="List of nodes in the mind map, each represented as a Node object with an 'id' and concise 'content' (no more than 5 words).",
        examples=[
            [
                Node(id="A", content="Fall of the Roman Empire"),
                Node(id="B", content="476 AD"),
                Node(id="C", content="Barbarian invasions"),
            ],
            [
                Node(id="A", content="Auxin is released"),
                Node(id="B", content="Travels to the roots"),
                Node(id="C", content="Root cells grow"),
            ],
        ],
    )
    edges: List[Edge] = Field(
        description="The edges connecting the nodes of the mind map, as a list of Edge objects with from_id and to_id fields representing the source and target node IDs.",
        examples=[
            [
                Edge(from_id="A", to_id="B"),
                Edge(from_id="A", to_id="C"),
                Edge(from_id="B", to_id="C"),
            ],
            [
                Edge(from_id="C", to_id="A"),
                Edge(from_id="B", to_id="C"),
                Edge(from_id="A", to_id="B"),
            ],
        ],
    )

    @model_validator(mode="after")
    def validate_mind_map(self) -> Self:
        all_nodes = [el.id for el in self.nodes]
        all_edges = [el.from_id for el in self.edges] + [el.to_id for el in self.edges]
        if set(all_nodes).issubset(set(all_edges)) and set(all_nodes) != set(all_edges):
            raise ValueError(
                "There are non-existing nodes listed as source or target in the edges"
            )
        return self


class MindMapCreationFailedWarning(Warning):
    """A warning returned if the mind map creation failed"""


class ClaimVerification(BaseModel):
    claim_is_true: bool = Field(
        description="Based on the provided sources information, the claim passes or not."
    )
    supporting_citations: Optional[List[str]] = Field(
        description="A minimum of one and a maximum of three citations from the sources supporting the claim. If the claim is not supported, please leave empty",
        default=None,
        min_length=1,
        max_length=3,
    )

    @model_validator(mode="after")
    def validate_claim_ver(self) -> Self:
        if not self.claim_is_true and self.supporting_citations is not None:
            self.supporting_citations = ["The claim was deemed false."]
        return self


# Instantiate ChromaDB storage globally
chroma_storage = ChromaStorage(
    persist_directory="./chroma_db",
    collection_name="notebookllama",
    embedding_model=OllamaEmbedding(model="nomic-embed-text")
)

# Use ChromaDB retriever and query engine
RETR = chroma_storage.get_retriever()
QE = chroma_storage.get_query_engine()

# Example: To add a document, use chroma_storage.add_text(text, metadata)
# Example: To query, use QE.query(question) or await QE.aquery(question)

llm_instance = OllamaLLM(model="gemma3:4b", temperature=0.1)
LLM_STRUCT = llm_instance.as_structured_llm(MindMap)
LLM_VERIFIER = llm_instance.as_structured_llm(ClaimVerification)


def md_table_to_pd_dataframe(md_table: Dict[str, list]) -> Optional[pd.DataFrame]:
    try:
        df = pd.DataFrame()
        for i in range(len(md_table["header"])):
            ls = [row[i] for row in md_table["rows"]]
            df[md_table["header"][i]] = ls
        return df
    except Exception as e:
        warnings.warn(f"Skipping table as an error occurred: {e}")
        return None


def rename_and_remove_past_images(path: str = "static/") -> List[str]:
    renamed = []
    if os.path.exists(path) and len(os.listdir(path)) >= 0:
        for image_file in os.listdir(path):
            image_path = os.path.join(path, image_file)
            if os.path.isfile(image_path) and "_at_" not in image_path:
                with open(image_path, "rb") as img:
                    bts = img.read()
                new_path = (
                    os.path.splitext(image_path)[0].replace("_current", "")
                    + f"_at_{datetime.now().strftime('%Y_%d_%m_%H_%M_%S_%f')[:-3]}.png"
                )
                with open(
                    new_path,
                    "wb",
                ) as img_tw:
                    img_tw.write(bts)
                renamed.append(new_path)
                os.remove(image_path)
    return renamed


def rename_and_remove_current_images(images: List[str]) -> List[str]:
    imgs = []
    for image in images:
        with open(image, "rb") as rb:
            bts = rb.read()
        with open(os.path.splitext(image)[0] + "_current.png", "wb") as wb:
            wb.write(bts)
        imgs.append(os.path.splitext(image)[0] + "_current.png")
        os.remove(image)
    return imgs


# Update parse_file and process_file to use local parser and ChromaDB
from src.notebookllama.local_parsers import LocalParser

local_parser = LocalParser()

async def parse_file(
    file_path: str, with_images: bool = False, with_tables: bool = False
) -> Union[Tuple[Optional[str], Optional[List[str]], Optional[List[pd.DataFrame]]]]:
    print(f"[parse_file] Start: file_path={file_path}")
    images: Optional[List[str]] = None
    text: Optional[str] = None
    tables: Optional[List[pd.DataFrame]] = None
    print("[parse_file] Before await local_parser.aparse")
    document = await local_parser.aparse(file_path=file_path)
    print("[parse_file] After await local_parser.aparse")
    text = document.text
    # (Images and tables logic can be added later if needed)
    print("[parse_file] Returning text, images, tables")
    return text, images, tables

async def process_file(
    filename: str,
) -> Union[Tuple[str, None], Tuple[None, None], Tuple[str, str]]:
    print(f"[process_file] Start: filename={filename}")
    text, _, _ = await parse_file(file_path=filename)
    print(f"[process_file] After parse_file: text is {'not None' if text is not None else 'None'}")
    if not text or not text.strip():
        print("[process_file] Skipping add_text: text is empty")
        return None, None
    # Add text to ChromaDB
    print("[process_file] Before chroma_storage.add_text")
    chroma_storage.add_text(text, metadata={"filename": filename})
    print("[process_file] After chroma_storage.add_text")
    print("[process_file] Returning (text, text)")
    return text, text


async def get_mind_map(summary: str, highlights: List[str]) -> Union[str, None]:
    try:
        keypoints = "\n- ".join(highlights)
        messages = [
            ChatMessage(
                role="user",
                content=f"This is the summary for my document: {summary}\n\nAnd these are the key points:\n- {keypoints}",
            )
        ]
        response = await LLM_STRUCT.achat(messages=messages)
        response_json = json.loads(response.message.content)
        net = Network(directed=True, height="750px", width="100%")
        net.set_options("""
            var options = {
            "physics": {
                "enabled": false
            }
            }
            """)
        nodes = response_json["nodes"]
        edges = response_json["edges"]
        for node in nodes:
            net.add_node(n_id=node["id"], label=node["content"])
        for edge in edges:
            net.add_edge(source=edge["from_id"], to=edge["to_id"])
        name = str(uuid.uuid4())
        net.save_graph(name + ".html")
        return name + ".html"
    except Exception as e:
        warnings.warn(
            message=f"An error occurred during the creation of the mind map: {e}",
            category=MindMapCreationFailedWarning,
        )
        return None


async def query_index(question: str) -> Union[str, None]:
    response = await QE.aquery(question)
    response = cast(Response, response)
    sources = []
    if not response.response:
        return None
    if response.source_nodes is not None:
        sources = [node.text for node in response.source_nodes]
    return (
        "## Answer\n\n"
        + response.response
        + "\n\n## Sources\n\n- "
        + "\n- ".join(sources)
    )


async def get_plots_and_tables(
    file_path: str,
) -> Union[Tuple[Optional[List[str]], Optional[List[pd.DataFrame]]]]:
    _, images, tables = await parse_file(
        file_path=file_path, with_images=True, with_tables=True
    )
    return images, tables


def verify_claim(
    claim: str,
    sources: str,
) -> Tuple[bool, Optional[List[str]]]:
    response = LLM_VERIFIER.chat(
        [
            ChatMessage(
                role="user",
                content=f"I have this claim: {claim} that is allegedgly supported by these sources:\n\n'''\n{sources}\n'''\n\nCan you please tell me whether or not this claim is thrutful and, if it is, identify one to three passages in the sources specifically supporting the claim?",
            )
        ]
    )
    response_json = json.loads(response.message.content)
    return response_json["claim_is_true"], response_json["supporting_citations"]
