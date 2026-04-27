"""
api/llm.py
Grounded answer generation via Groq with exponential-backoff retry.
"""

import time
import logging

from groq import Groq, RateLimitError, APITimeoutError, APIStatusError

from config import settings

logger = logging.getLogger("documind.llm")


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Generate an answer grounded in retrieved chunks.
    Retries up to settings.GROQ_MAX_RETRIES times on rate-limit / timeout errors.
    """
    client = Groq()

    # Build labeled context blocks
    parts: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        meta  = chunk["metadata"]
        label = f"[Source {i}: {meta['file_name']}"
        if meta.get("page"):
            label += f", page {meta['page']}"
        label += "]"
        parts.append(f"{label}\n{chunk['text']}")

    context = "\n\n---\n\n".join(parts)

    prompt = f"""You are DocuMind AI, a helpful document assistant. The user has uploaded their documents and wants to ask questions about them.

Below are the most relevant excerpts retrieved from the user's documents:

--- Document Excerpts ---
{context}
--- End of Excerpts ---

Question: {query}

Instructions:
1. If the excerpts directly answer the question, answer using ONLY the document content and cite sources (e.g. "According to 4.Quadratic Equations 2020.pdf...").
2. If the excerpts contain PRACTICE PROBLEMS or EXERCISES on the topic (but not a formal definition), do the following:
   - First, provide a clear explanation of the concept from your knowledge.
   - Then say: "Your document '{{filename}}' contains practice problems on this topic:" and list 1-2 example problems from the excerpts.
3. If the excerpts are completely unrelated to the question, say: "I couldn't find information about this in your uploaded documents."
4. Always be helpful, clear, and concise.

Answer:"""

    last_error: Exception | None = None

    for attempt in range(1, settings.GROQ_MAX_RETRIES + 1):
        try:
            logger.info("Groq request attempt %d/%d", attempt, settings.GROQ_MAX_RETRIES)
            response = client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS,
            )
            return response.choices[0].message.content

        except (RateLimitError, APITimeoutError) as e:
            last_error = e
            wait = 2 ** attempt   # exponential back-off: 2s, 4s, 8s …
            logger.warning("Groq transient error (%s). Retrying in %ds…", type(e).__name__, wait)
            time.sleep(wait)

        except APIStatusError as e:
            # Non-retryable API error (e.g. bad request, auth error)
            logger.error("Groq API error %s: %s", e.status_code, e.message)
            raise

    logger.error("Groq request failed after %d attempts.", settings.GROQ_MAX_RETRIES)
    raise RuntimeError(
        f"LLM unavailable after {settings.GROQ_MAX_RETRIES} retries: {last_error}"
    )
