from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Iterable, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.genai import errors as google_genai_errors

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
FALLBACK_CHAT_MODEL = "gemini-2.5-flash"


def configure_logging(enabled: bool | None = None) -> None:
	"""Configure application logging from environment or an explicit flag."""

	if enabled is None:
		enabled = os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() in {"1", "true", "yes", "on"}

	level = logging.DEBUG if enabled else logging.WARNING
	logging.basicConfig(
		level=level,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)
	logger.debug("Logging configured: enabled=%s level=%s", enabled, logging.getLevelName(level))


@lru_cache(maxsize=1)
def get_chat_model(model: str = DEFAULT_CHAT_MODEL) -> ChatGoogleGenerativeAI:
	"""Return a cached Gemini chat model wrapped by LangChain."""

	logger.debug("Creating Gemini chat client with model=%s", model)
	return ChatGoogleGenerativeAI(model=model, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> GoogleGenerativeAIEmbeddings:
	"""Return a cached Gemini embedding model wrapped by LangChain."""

	logger.debug("Creating Gemini embedding client with model=%s", model)
	return GoogleGenerativeAIEmbeddings(model=model)


def embed_text(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
	"""Embed a single text string with Gemini."""

	logger.debug("Embedding single text: chars=%d model=%s", len(text), model)
	return get_embeddings(model).embed_query(text)


def embed_texts(texts: Sequence[str], model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
	"""Embed multiple text strings with Gemini."""

	logger.debug("Embedding batch: count=%d model=%s", len(texts), model)
	return get_embeddings(model).embed_documents(list(texts))


def build_rag_prompt() -> ChatPromptTemplate:
	"""Prompt used to answer questions from retrieved context."""

	return ChatPromptTemplate.from_messages(
		[
			(
				"system",
				"You answer using only the provided PDF context. If the answer is not explicitly supported by the context, respond exactly with: Not found in PDF.",
			),
			("human", "Question: {question}\n\nContext:\n{context}"),
		]
	)


def _is_model_not_found_error(error: Exception) -> bool:
	status_code = getattr(error, "status_code", None)
	if status_code == 404:
		return True

	message = str(error).lower()
	return "not found" in message and "model" in message


def _invoke_chat_with_fallback(invoke_callable, model: str, fallback_model: str):
	try:
		return invoke_callable(model)
	except Exception as error:
		if model != fallback_model and _is_model_not_found_error(error):
			logger.warning("Gemini model %s is unavailable; falling back to %s", model, fallback_model)
			return invoke_callable(fallback_model)
		raise


def format_context(documents: Iterable[Document]) -> str:
	"""Convert retrieved documents into a compact prompt context."""

	parts: list[str] = []
	for index, document in enumerate(documents, start=1):
		source = document.metadata.get("source", "unknown")
		category = document.metadata.get("category", "unknown")
		parts.append(f"[{index}] source={source} category={category}\n{document.page_content}")
	return "\n\n".join(parts)


def generate_response(question: str, documents: Sequence[Document]) -> str:
	"""Generate a grounded Gemini response from retrieved documents."""

	logger.debug("Generating grounded response: question_chars=%d documents=%d", len(question), len(documents))
	prompt = build_rag_prompt()

	def invoke_for_model(model_name: str):
		chain = prompt | get_chat_model(model_name)
		return chain.invoke({"question": question, "context": format_context(documents)})

	result = _invoke_chat_with_fallback(invoke_for_model, DEFAULT_CHAT_MODEL, FALLBACK_CHAT_MODEL)
	logger.debug("Generated response chars=%d", len(result.content))
	return result.content


def invoke_gemini(prompt_text: str) -> str:
	"""Simple one-shot Gemini text generation helper."""

	logger.debug("Invoking Gemini directly: prompt_chars=%d", len(prompt_text))

	def invoke_for_model(model_name: str):
		return get_chat_model(model_name).invoke([HumanMessage(content=prompt_text)])

	response = _invoke_chat_with_fallback(invoke_for_model, DEFAULT_CHAT_MODEL, FALLBACK_CHAT_MODEL)
	logger.debug("Direct Gemini response chars=%d", len(response.content))
	return response.content
