/**
 * Embedding generation using Ollama API.
 *
 * Wraps the Ollama /api/embed endpoint for generating semantic embeddings.
 * Replaces the original @huggingface/transformers implementation with
 * a lightweight HTTP-based approach.
 *
 * Default model: nomic-embed-text-v2-moe (768 dimensions)
 */

import { EMBEDDING_CONFIG, type Embedding } from "./types";
import { logger } from "./logger";

/**
 * Generate an embedding vector for the given text via Ollama.
 *
 * @param text - The text to embed.
 * @returns Promise resolving to a 768-dimensional Float32Array.
 * @throws Error if Ollama is unreachable or embedding fails.
 */
export async function embed(text: string): Promise<Embedding> {
	const startTime = performance.now();

	const response = await fetch(`${EMBEDDING_CONFIG.OLLAMA_URL}/api/embed`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			model: EMBEDDING_CONFIG.MODEL,
			input: text,
		}),
	});

	if (!response.ok) {
		throw new Error(
			`Ollama embed API error: ${response.status} ${response.statusText}`,
		);
	}

	const data = (await response.json()) as {
		embeddings: number[][];
	};

	const duration = performance.now() - startTime;

	if (!data.embeddings || data.embeddings.length === 0) {
		throw new Error("Ollama returned empty embeddings");
	}

	const embedding = new Float32Array(data.embeddings[0]);

	// Validate dimension
	if (embedding.length !== EMBEDDING_CONFIG.DIMENSION) {
		logger.warn("Embedding dimension mismatch", {
			expected: EMBEDDING_CONFIG.DIMENSION,
			actual: embedding.length,
		});
		// Resize if necessary
		if (embedding.length > EMBEDDING_CONFIG.DIMENSION) {
			return embedding.slice(0, EMBEDDING_CONFIG.DIMENSION);
		} else {
			const padded = new Float32Array(EMBEDDING_CONFIG.DIMENSION);
			padded.set(embedding);
			return padded;
		}
	}

	logger.debug("Embedding generated", {
		textLength: text.length,
		duration: Math.round(duration * 100) / 100,
	});

	return embedding;
}

/**
 * Generate embeddings for multiple texts in batch via Ollama.
 * Ollama's /api/embed accepts an array of inputs natively.
 *
 * @param texts - Array of texts to embed.
 * @returns Promise resolving to array of embeddings.
 */
export async function embedBatch(texts: string[]): Promise<Embedding[]> {
	const startTime = performance.now();

	const response = await fetch(`${EMBEDDING_CONFIG.OLLAMA_URL}/api/embed`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			model: EMBEDDING_CONFIG.MODEL,
			input: texts,
		}),
	});

	if (!response.ok) {
		throw new Error(
			`Ollama embed API error: ${response.status} ${response.statusText}`,
		);
	}

	const data = (await response.json()) as {
		embeddings: number[][];
	};

	const embeddings: Embedding[] = data.embeddings.map((vec) => {
		const embedding = new Float32Array(vec);
		// Resize if dimension doesn't match
		if (embedding.length !== EMBEDDING_CONFIG.DIMENSION) {
			if (embedding.length > EMBEDDING_CONFIG.DIMENSION) {
				return embedding.slice(0, EMBEDDING_CONFIG.DIMENSION);
			} else {
				const padded = new Float32Array(EMBEDDING_CONFIG.DIMENSION);
				padded.set(embedding);
				return padded;
			}
		}
		return embedding;
	});

	const duration = performance.now() - startTime;

	logger.debug("Batch embeddings generated", {
		count: texts.length,
		duration: Math.round(duration * 100) / 100,
	});

	return embeddings;
}

/**
 * Check if embedding is ready (Ollama is always "ready" if reachable).
 */
export function isEmbedReady(): boolean {
	return true;
}

/**
 * No-op — Ollama manages its own model lifecycle.
 */
export async function unloadEmbed(): Promise<void> {
	logger.debug("unloadEmbed: no-op for Ollama backend");
}
