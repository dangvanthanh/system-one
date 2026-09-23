import assert from "node:assert/strict";
import { Hono } from "hono";
export type Direction = "UP" | "RIGHT" | "DOWN" | "LEFT";
export type LegalMove = {
	direction: Direction;
	board: number[][];
	gainedScore: number;
};
export type DecisionRequest = {
	board: number[][];
	score: number;
	legalMoves: LegalMove[];
};

type ChoiceQuestion = {
	type: "choice";
	instructions: string;
	criteria: string[];
};
type DecisionClient = {
	systemOne(state: unknown, questions: DecisionQuestions): Promise<unknown>;
};
type DecisionQuestions = { move: ChoiceQuestion };
export type HandlerOptions = {
	apiKey?: string;
	upstreamFetch?: typeof fetch;
	getLaya?: () => Promise<DecisionClient>;
};

const DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"] as const;
const MAX_REQUEST_CHARS = 16_384;
const SEMIF_PORT = Number(Bun.env.SEMIF_SERVER_PORT ?? 38081);
const SEMIF_BASE_URL = `http://127.0.0.1:${SEMIF_PORT}`;
const SEMIF_MODEL = "MiniCPM5-2B-Q4_K_M.gguf";
const SEMIF_LABELS = ["A", "B", "C", "D"] as const;
let semifProcess: Bun.Subprocess | null = null;
let semifLoading: Promise<void> | null = null;
let semifLabelIds: number[] | null = null;
let semifApiKey: string | null = null;

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isDirection(value: unknown): value is Direction {
	return DIRECTIONS.includes(value as Direction);
}

function isNonNegativeInteger(value: unknown): value is number {
	return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isTile(value: unknown): value is number {
	return isNonNegativeInteger(value) && (value === 0 || Number.isInteger(Math.log2(value)));
}

function isBoard(value: unknown): value is number[][] {
	return (
		Array.isArray(value) &&
		value.length === 4 &&
		value.every((row) => Array.isArray(row) && row.length === 4 && row.every(isTile))
	);
}

function parseDecisionRequest(value: unknown): DecisionRequest | null {
	if (!isRecord(value) || !isBoard(value.board)) return null;
	if (!isNonNegativeInteger(value.score) || !Array.isArray(value.legalMoves)) {
		return null;
	}
	if (value.legalMoves.length > 4) return null;

	const legalMoves: LegalMove[] = [];
	const seen = new Set<Direction>();
	for (const move of value.legalMoves) {
		if (
			!isRecord(move) ||
			!isDirection(move.direction) ||
			!isBoard(move.board) ||
			!isNonNegativeInteger(move.gainedScore) ||
			seen.has(move.direction)
		) {
			return null;
		}
		seen.add(move.direction);
		legalMoves.push({
			direction: move.direction,
			board: move.board,
			gainedScore: move.gainedScore,
		});
	}

	return { board: value.board, score: value.score, legalMoves };
}

function buildDecision(body: DecisionRequest) {
	const questions: DecisionQuestions = {
		move: {
			type: "choice",
			instructions:
				"Choose the legal move most likely to maximize long-term score and reach 2048. Use `legalMoves` outcomes; prefer mobility, merge potential, monotonic rows or columns, and keeping the highest tile in a corner.",
			criteria: body.legalMoves.map(({ direction }) => direction),
		},
	};
	return {
		state: {
			game: "2048",
			objective:
				"Reach the highest tile and maximize long-term score without running out of legal moves.",
			board: body.board,
			score: body.score,
			legalMoves: body.legalMoves,
		},
		questions,
	};
}

function normalizeDecision(
	result: unknown,
	legalMoves: LegalMove[],
): { move: Direction; confidence: number; probabilities: Record<string, unknown> } | null {
	const legalDirections = new Set(legalMoves.map(({ direction }) => direction));
	const answer =
		isRecord(result) && isRecord(result.answers) && isRecord(result.answers.move)
			? result.answers.move
			: null;
	const probabilities = answer?.probabilities;
	const entries = isRecord(probabilities) ? Object.entries(probabilities) : null;
	const validEntry = (entry: [string, unknown]) =>
		legalDirections.has(entry[0] as Direction) &&
		typeof entry[1] === "number" &&
		Number.isFinite(entry[1]) &&
		entry[1] >= 0 &&
		entry[1] <= 1;
	const validProbabilities =
		entries !== null &&
		entries.length === legalDirections.size &&
		entries.every(validEntry) &&
		Math.abs(entries.reduce((sum, [, probability]) => sum + (probability as number), 0) - 1) <=
			entries.length * 0.005 + Number.EPSILON;

	if (
		!answer ||
		answer.type !== "choice" ||
		!isDirection(answer.choice) ||
		!legalDirections.has(answer.choice) ||
		!validProbabilities
	) {
		return null;
	}
	const confidence =
		typeof answer.confidence === "number" &&
		Number.isFinite(answer.confidence) &&
		answer.confidence >= 0 &&
		answer.confidence <= 1
			? answer.confidence
			: (probabilities as Record<string, unknown>)[answer.choice];
	if (
		typeof confidence !== "number" ||
		!Number.isFinite(confidence) ||
		confidence < 0 ||
		confidence > 1
	) {
		return null;
	}

	return {
		move: answer.choice,
		confidence,
		probabilities: probabilities as Record<string, unknown>,
	};
}

function json(body: object, status = 200): Response {
	return Response.json(body, { status });
}

async function readDecisionRequest(request: Request): Promise<DecisionRequest | Response> {
	if (!request.headers.get("content-type")?.startsWith("application/json")) {
		return json({ error: "Expected application/json" }, 415);
	}

	let text: string;
	try {
		text = await request.text();
	} catch {
		return json({ error: "Invalid decision request" }, 400);
	}
	if (text.length > MAX_REQUEST_CHARS) return json({ error: "Request too large" }, 413);

	let value: unknown;
	try {
		value = JSON.parse(text);
	} catch {
		return json({ error: "Invalid decision request" }, 400);
	}
	const body = parseDecisionRequest(value);
	return body ?? json({ error: "Invalid decision request" }, 400);
}

function slideRow(row: number[]): number[] {
	const values = row.filter((tile) => tile !== 0);
	const next: number[] = [];
	for (let index = 0; index < values.length; index++) {
		if (values[index] === values[index + 1]) {
			next.push(values[index] * 2);
			index++;
		} else {
			next.push(values[index]);
		}
	}
	while (next.length < 4) next.push(0);
	return next;
}

function moveChangesBoard(board: number[][], direction: Direction): boolean {
	const rotated =
		direction === "UP"
			? board[0].map((_, column) => board.map((row) => row[column])).reverse()
			: direction === "DOWN"
				? board[0]
						.map((_, column) => board.map((row) => row[column]).reverse())
						.map((row) => [...row].reverse())
				: direction === "RIGHT"
					? board.map((row) => [...row].reverse())
					: board;
	return rotated.some((row, index) => slideRow(row).join(",") !== rotated[index].join(","));
}

export function isGameOver(board: number[][]): boolean {
	return (DIRECTIONS as readonly Direction[]).every(
		(direction) => !moveChangesBoard(board, direction),
	);
}

let layaBridgeChild: Bun.Subprocess | null = null;

function stopLayaBridge(): void {
	layaBridgeChild?.kill();
	layaBridgeChild = null;
}

async function spawnLayaBridge(): Promise<DecisionClient> {
	const child = Bun.spawn(["python3", new URL("./laya_bridge.py", import.meta.url).pathname], {
		stdin: "pipe",
		stdout: "pipe",
		stderr: "inherit",
	});
	layaBridgeChild = child;

	type Pending = { resolve: (value: unknown) => void; reject: (error: Error) => void };
	const pending = new Map<number, Pending>();
	let nextId = 0;
	let buffer = "";
	const decoder = new TextDecoder();

	const failAll = (message: string) => {
		for (const { reject } of pending.values()) reject(new Error(message));
		pending.clear();
	};

	void (async () => {
		const reader = child.stdout.getReader();
		for (;;) {
			const { done, value } = await reader.read();
			if (done) return failAll("Laya Python sidecar closed unexpectedly");
			buffer += decoder.decode(value, { stream: true });
			let newline: number;
			while ((newline = buffer.indexOf("\n")) !== -1) {
				const line = buffer.slice(0, newline).trim();
				buffer = buffer.slice(newline + 1);
				if (!line) continue;
				try {
					const message = JSON.parse(line) as { id?: number; error?: string };
					const id = message.id;
					if (id === undefined) continue;
					const waiter = pending.get(id);
					if (!waiter) continue;
					pending.delete(id);
					if (message.error !== undefined) waiter.reject(new Error(message.error));
					else waiter.resolve(message);
				} catch {
					// Ignore malformed sidecar lines.
				}
			}
		}
	})();
	child.exited.then((code) => failAll(`Laya Python sidecar exited (code ${code})`));

	return {
		systemOne: (state, questions) =>
			new Promise((resolve, reject) => {
				const id = nextId++;
				pending.set(id, { resolve, reject });
				try {
					child.stdin.write(`${JSON.stringify({ id, state, questions })}\n`);
				} catch (error) {
					pending.delete(id);
					reject(error instanceof Error ? error : new Error("Laya sidecar write failed"));
				}
			}),
	};
}

export function createLayaLoader(
	factory: () => Promise<DecisionClient> = spawnLayaBridge,
): () => Promise<DecisionClient> {
	let pending: Promise<DecisionClient> | null = null;
	return async () => {
		if (!pending) {
			pending = factory().catch((error) => {
				pending = null;
				throw error;
			});
		}
		return pending;
	};
}

const getSharedLaya = createLayaLoader();

async function isSemifReady(): Promise<boolean> {
	if (!semifProcess || semifProcess.exitCode !== null) return false;
	try {
		const response = await fetch(`${SEMIF_BASE_URL}/health`, {
			headers: { authorization: `Bearer ${semifApiKey}` },
			signal: AbortSignal.timeout(1000),
		});
		return response.ok;
	} catch {
		return false;
	}
}

async function ensureSemifServer(): Promise<void> {
	if (await isSemifReady()) return;
	if (semifLoading) return semifLoading;

	const startup = (async () => {
		semifProcess?.kill();
		semifProcess = null;
		semifLabelIds = null;
		semifApiKey = crypto.randomUUID();
		const executable = Bun.env.LLAMA_SERVER_BIN ?? "llama-server";
		try {
			semifProcess = Bun.spawn(
				[
					executable,
					"--hf-repo",
					"openbmb/MiniCPM5-2B-GGUF:Q4_K_M",
					"--hf-file",
					SEMIF_MODEL,
					"--no-mmproj",
					"--host",
					"127.0.0.1",
					"--port",
					String(SEMIF_PORT),
					"--no-ui",
					"--n-gpu-layers",
					"all",
					"--ctx-size",
					"2048",
					"--api-key",
					semifApiKey,
				],
				{ stdin: "ignore", stdout: "inherit", stderr: "inherit" },
			);
		} catch {
			throw new Error(
				"Could not start local llama-server. Install llama.cpp or set LLAMA_SERVER_BIN.",
			);
		}

		const child = semifProcess;
		const deadline = Date.now() + 15 * 60_000;
		while (Date.now() < deadline) {
			if (child.exitCode !== null) {
				semifProcess = null;
				throw new Error(
					`Local llama-server exited during SemIf model load (code ${child.exitCode}); see the server terminal for details.`,
				);
			}
			if (await isSemifReady()) return;
			await Bun.sleep(250);
		}
		child.kill();
		semifProcess = null;
		throw new Error("SemIf model load timed out after 15 minutes; check server logs.");
	})();
	semifLoading = startup;
	try {
		await startup;
	} catch (error) {
		semifProcess?.kill();
		semifProcess = null;
		throw error;
	} finally {
		if (semifLoading === startup) semifLoading = null;
	}
}

function stopSemifServer(): void {
	semifProcess?.kill();
	semifProcess = null;
}

function sampledOptionLogprob(value: unknown, label: string): number | null {
	if (!isRecord(value) || !Array.isArray(value.choices) || !isRecord(value.choices[0])) return null;
	const logprobs = value.choices[0].logprobs;
	if (!isRecord(logprobs) || !Array.isArray(logprobs.content) || !isRecord(logprobs.content[0]))
		return null;
	const token = logprobs.content[0];
	const tokenText = typeof token.token === "string" ? token.token.trim() : "";
	const bytes = Array.isArray(token.bytes) ? token.bytes : [];
	const ascii = label.charCodeAt(0);
	const matches =
		tokenText === label ||
		(bytes.length === 1 && bytes[0] === ascii) ||
		(bytes.length === 2 && bytes[0] === 32 && bytes[1] === ascii);
	return matches && typeof token.logprob === "number" && Number.isFinite(token.logprob)
		? token.logprob
		: null;
}

function softmax(values: number[]): number[] {
	const maximum = Math.max(...values);
	const exponents = values.map((value) => Math.exp(value - maximum));
	const total = exponents.reduce((sum, value) => sum + value, 0);
	return exponents.map((value) => value / total);
}

export function createHandler({
	apiKey,
	upstreamFetch = fetch,
	getLaya = getSharedLaya,
}: HandlerOptions = {}): (request: Request) => Promise<Response> {
	const app = new Hono();

	const asset = (path: string, type: string) =>
		new Response(Bun.file(new URL(path, import.meta.url)), {
			headers: { "content-type": type },
		});

	app.get("/", () => asset("./index.html", "text/html; charset=utf-8"));
	app.get("/api/semif/status", async () =>
		json({ ready: await isSemifReady(), loading: Boolean(semifLoading) }),
	);
	app.post("/api/semif/load", async () => {
		try {
			await ensureSemifServer();
			return json({ ready: true });
		} catch (error) {
			return json(
				{ error: error instanceof Error ? error.message : "SemIf model load failed" },
				503,
			);
		}
	});
	app.post("/api/semif", async (c) => {
		const parsed = await readDecisionRequest(c.req.raw);
		if (parsed instanceof Response) return parsed;
		const body = parsed;
		if (body.legalMoves.length === 0 || isGameOver(body.board)) {
			return json({ gameOver: true });
		}
		if (!(await isSemifReady())) {
			return json({ error: "SemIf model is not loaded. Use Load SemIf model." }, 503);
		}

		try {
			if (!semifLabelIds) {
				const tokenized = await Promise.all(
					SEMIF_LABELS.map(async (label) => {
						const response = await fetch(`${SEMIF_BASE_URL}/tokenize`, {
							method: "POST",
							headers: {
								"content-type": "application/json",
								authorization: `Bearer ${semifApiKey}`,
							},
							body: JSON.stringify({ content: label }),
							signal: c.req.raw.signal,
						});
						const result = await response.json();
						if (
							!response.ok ||
							!isRecord(result) ||
							!Array.isArray(result.tokens) ||
							result.tokens.length !== 1
						) {
							throw new Error("Local tokenizer did not resolve option letters.");
						}
						return result.tokens[0] as number;
					}),
				);
				if (tokenized.some((token) => !Number.isInteger(token) || token < 0)) {
					return json({ error: "Local tokenizer returned invalid option ids" }, 502);
				}
				semifLabelIds = tokenized;
			}

			const labels = SEMIF_LABELS.slice(0, body.legalMoves.length);
			const messages = [
				{
					role: "system",
					content:
						"Make the requested decision from the supplied state. Follow the output format exactly.",
				},
				{
					role: "user",
					content: `State:\n${JSON.stringify({ game: "2048", board: body.board, score: body.score })}\n\nQuestion:\nWhich legal move should be played next to maximize long-term score and reach 2048?\n\nAllowed options:\n${body.legalMoves
						.map(
							({ direction, board, gainedScore }, index) =>
								`${labels[index]}. resulting board ${board.map((row) => row.join(",")).join(" / ")}; immediate score gain ${gainedScore}`,
						)
						.join("\n")}\n\nReply with exactly one option letter from: ${labels.join(", ")}.`,
				},
			];
			const logits: number[] = [];
			for (let index = 0; index < labels.length; index++) {
				const response = await fetch(`${SEMIF_BASE_URL}/v1/chat/completions`, {
					method: "POST",
					headers: {
						"content-type": "application/json",
						authorization: `Bearer ${semifApiKey}`,
					},
					body: JSON.stringify({
						model: SEMIF_MODEL,
						messages,
						max_tokens: 1,
						temperature: 1,
						top_k: 0,
						top_p: 1,
						logprobs: true,
						top_logprobs: 8,
						logit_bias: { [String(semifLabelIds[index])]: 100 },
						chat_template_kwargs: { enable_thinking: false },
						reasoning_effort: "none",
					}),
					signal: c.req.raw.signal,
				});
				const result = await response.json();
				const logprob = response.ok ? sampledOptionLogprob(result, labels[index]) : null;
				if (logprob === null) {
					throw new Error("Local SemIf inference returned an invalid option score.");
				}
				logits.push(logprob);
			}

			const probabilities = softmax(logits);
			const best = probabilities.indexOf(Math.max(...probabilities));
			return json({
				move: body.legalMoves[best].direction,
				confidence: probabilities[best],
				probabilities: Object.fromEntries(
					body.legalMoves.map(({ direction }, index) => [direction, probabilities[index]]),
				),
				gameOver: false,
			});
		} catch (error) {
			return json(
				{ error: error instanceof Error ? error.message : "SemIf local inference failed" },
				502,
			);
		}
	});

	app.post("/api/jev", async (c) => {
		const parsed = await readDecisionRequest(c.req.raw);
		if (parsed instanceof Response) return parsed;
		const body = parsed;
		if (body.legalMoves.length === 0 || isGameOver(body.board)) {
			return json({ gameOver: true });
		}
		if (!apiKey?.trim()) {
			return json({ error: "Command Code API key is not configured" }, 503);
		}

		const { state, questions } = buildDecision(body);
		let result: unknown;
		try {
			const upstream = await upstreamFetch("https://api.commandcode.ai/provider/v1/systemone", {
				method: "POST",
				headers: {
					authorization: `Bearer ${apiKey.trim()}`,
					"content-type": "application/json",
				},
				body: JSON.stringify({ model: "typesafe/jev", state, questions }),
			});
			if (!upstream.ok) return json({ error: "Jev request failed" }, 502);
			result = await upstream.json();
		} catch {
			return json({ error: "Jev request failed" }, 502);
		}

		const decision = normalizeDecision(result, body.legalMoves);
		if (!decision) {
			return json({ error: "Jev returned an invalid decision" }, 502);
		}
		return json({ ...decision, gameOver: false });
	});

	app.post("/api/laya", async (c) => {
		const parsed = await readDecisionRequest(c.req.raw);
		if (parsed instanceof Response) return parsed;
		const body = parsed;
		if (body.legalMoves.length === 0 || isGameOver(body.board)) {
			return json({ gameOver: true });
		}

		const { state, questions } = buildDecision(body);
		let result: unknown;
		try {
			result = await (await getLaya()).systemOne(state, questions);
		} catch {
			return json({ error: "Laya request failed" }, 502);
		}

		const decision = normalizeDecision(result, body.legalMoves);
		if (!decision) {
			return json({ error: "Laya returned an invalid decision" }, 502);
		}
		return json({ ...decision, gameOver: false });
	});

	app.notFound((c) => json({ error: "Not found" }, 404));

	return async (request) => app.fetch(request);
}

async function selfCheck() {
	const validBody: DecisionRequest = {
		board: [
			[2, 2, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0],
		],
		score: 0,
		legalMoves: [
			{
				direction: "LEFT",
				board: [
					[4, 0, 0, 0],
					[0, 0, 0, 0],
					[0, 0, 0, 0],
					[0, 0, 0, 0],
				],
				gainedScore: 4,
			},
		],
	};
	const post = (body: unknown, contentType = "application/json", path = "/api/laya") =>
		new Request(`http://localhost${path}`, {
			method: "POST",
			headers: { "content-type": contentType },
			body: typeof body === "string" ? body : JSON.stringify(body),
		});
	for (const invalid of [
		"{",
		{ ...validBody, board: [[3]] },
		{ ...validBody, score: -1 },
		{
			...validBody,
			legalMoves: [{ ...validBody.legalMoves[0], direction: "JUMP" }],
		},
		{
			...validBody,
			legalMoves: [...validBody.legalMoves, validBody.legalMoves[0]],
		},
	]) {
		const response = await createHandler()(post(invalid));
		assert.equal(response.status, 400);
	}
	const validationHandler = createHandler();
	assert.equal((await validationHandler(post(validBody, "text/plain"))).status, 415);
	assert.equal((await validationHandler(post("x".repeat(16_385)))).status, 413);

	const fakeAnswer = {
		model: "test",
		answers: {
			move: {
				type: "choice" as const,
				choice: "LEFT",
				confidence: 1,
				probabilities: { LEFT: 1 },
				rl_agent: { act_probability: 1 },
			},
		},
		usage: { input_tokens: 1, output_tokens: 0 },
	};
	let layaState: unknown;
	let layaQuestions: unknown;
	let layaCalls = 0;
	const fakeLaya: DecisionClient = {
		async systemOne(state, questions) {
			layaCalls++;
			layaState = state;
			layaQuestions = questions;
			return fakeAnswer;
		},
	};
	const missingKey = await createHandler({ getLaya: async () => fakeLaya })(
		post(validBody, "application/json", "/api/jev"),
	);
	assert.equal(missingKey.status, 503);
	assert.deepEqual(await missingKey.json(), {
		error: "Command Code API key is not configured",
	});

	let jevUrl = "";
	let jevHeaders: HeadersInit | undefined;
	let jevPayload: any;
	const commandUpstream = (async (url: RequestInfo | URL, init?: RequestInit) => {
		jevUrl = String(url);
		jevHeaders = init?.headers;
		jevPayload = JSON.parse(String(init?.body));
		return Response.json(fakeAnswer);
	}) as unknown as typeof fetch;

	const handler = createHandler({
		apiKey: "secret",
		upstreamFetch: commandUpstream,
		getLaya: async () => fakeLaya,
	});
	for (const path of ["/api/jev", "/api/laya"]) {
		const response = await handler(post(validBody, "application/json", path));
		assert.equal(response.status, 200);
		assert.deepEqual(await response.json(), {
			move: "LEFT",
			confidence: 1,
			probabilities: { LEFT: 1 },
			gameOver: false,
		});
	}
	assert.equal(jevUrl, "https://api.commandcode.ai/provider/v1/systemone");
	assert.deepEqual(jevHeaders, {
		authorization: "Bearer secret",
		"content-type": "application/json",
	});
	assert.equal(jevPayload.model, "typesafe/jev");
	assert.equal(jevPayload.questions.move.type, "choice");
	assert.deepEqual(jevPayload.questions.move.criteria, ["LEFT"]);

	assert.equal(isGameOver(validBody.board), false);
	const deadBoard: DecisionRequest = {
		...validBody,
		board: [
			[2, 4, 2, 4],
			[4, 2, 4, 2],
			[2, 4, 2, 4],
			[4, 2, 4, 2],
		],
	};
	assert.equal(isGameOver(deadBoard.board), true);
	const deadResponse = await handler(
		post({ ...deadBoard, legalMoves: [] }, "application/json", "/api/laya"),
	);
	assert.equal(deadResponse.status, 200);
	assert.deepEqual(await deadResponse.json(), { gameOver: true });
	assert.equal(layaCalls, 1);
	assert.deepEqual((layaState as { legalMoves: unknown }).legalMoves, validBody.legalMoves);
	assert.deepEqual((layaQuestions as { move: { criteria: unknown } }).move.criteria, ["LEFT"]);

	const roundedMoves: LegalMove[] = [
		validBody.legalMoves[0],
		{ ...validBody.legalMoves[0], direction: "RIGHT" },
		{ ...validBody.legalMoves[0], direction: "DOWN" },
	];
	assert.ok(
		normalizeDecision(
			{
				answers: {
					move: {
						type: "choice",
						choice: "LEFT",
						confidence: 0.1,
						probabilities: { LEFT: 0.33, RIGHT: 0.33, DOWN: 0.33 },
					},
				},
			},
			roundedMoves,
		),
	);

	const threeMoves = [
		validBody.legalMoves[0],
		{ ...validBody.legalMoves[0], direction: "RIGHT" as const },
		{ ...validBody.legalMoves[0], direction: "DOWN" as const },
	];
	const noConfidence = {
		...fakeAnswer,
		answers: {
			move: {
				type: "choice",
				choice: "LEFT",
				probabilities: { LEFT: 0.5, RIGHT: 0.3, DOWN: 0.2 },
			},
		},
	};
	const fallbackHandler = createHandler({
		apiKey: "secret",
		upstreamFetch: (async () => Response.json(noConfidence)) as unknown as typeof fetch,
		getLaya: async () => ({
			async systemOne() {
				return noConfidence;
			},
		}),
	});
	const fallback = await fallbackHandler(
		post({ ...validBody, legalMoves: threeMoves }, "application/json", "/api/jev"),
	);
	assert.equal(fallback.status, 200);
	assert.equal((await fallback.json()).confidence, 0.5);

	const invalidAnswer = {
		...fakeAnswer,
		answers: {
			move: { ...fakeAnswer.answers.move, probabilities: {} },
		},
	};
	for (const [path, expectedError, options] of [
		[
			"/api/jev",
			"Jev returned an invalid decision",
			{
				apiKey: "secret",
				upstreamFetch: (async () => Response.json(invalidAnswer)) as unknown as typeof fetch,
			},
		],
		[
			"/api/laya",
			"Laya returned an invalid decision",
			{
				getLaya: async () => ({
					async systemOne() {
						return invalidAnswer;
					},
				}),
			},
		],
	] as const) {
		const invalidHandler = createHandler(options as HandlerOptions);
		const response = await invalidHandler(post(validBody, "application/json", path));
		assert.equal(response.status, 502);
		assert.deepEqual(await response.json(), { error: expectedError });
	}

	for (const [path, expectedError, options] of [
		[
			"/api/jev",
			"Jev request failed",
			{
				apiKey: "secret",
				upstreamFetch: (async () => {
					throw new Error("private");
				}) as unknown as typeof fetch,
			},
		],
		[
			"/api/laya",
			"Laya request failed",
			{
				getLaya: async () => {
					throw new Error("private");
				},
			},
		],
	] as const) {
		const response = await createHandler(options as HandlerOptions)(
			post(validBody, "application/json", path),
		);
		assert.equal(response.status, 502);
		assert.deepEqual(await response.json(), { error: expectedError });
	}

	let loads = 0;
	const loader = createLayaLoader(async () => {
		loads++;
		await Promise.resolve();
		return fakeLaya;
	});
	const [first, second] = await Promise.all([loader(), loader()]);
	assert.equal(first, fakeLaya);
	assert.equal(second, fakeLaya);
	assert.equal(loads, 1);

	let attempts = 0;
	const retryingLoader = createLayaLoader(async () => {
		attempts++;
		if (attempts === 1) throw new Error("first load failed");
		return fakeLaya;
	});
	await assert.rejects(retryingLoader(), /first load failed/);
	assert.equal(await retryingLoader(), fakeLaya);
	assert.equal(attempts, 2);
}

if (import.meta.main) {
	if (Bun.argv.includes("--self-check")) {
		await selfCheck();
		console.log("server self-check passed");
	} else {
		process.on("exit", () => {
			stopLayaBridge();
			stopSemifServer();
		});
		process.once("SIGINT", () => {
			stopSemifServer();
			process.exit(130);
		});
		process.once("SIGTERM", () => {
			stopSemifServer();
			process.exit(143);
		});
		const port = Number(Bun.env.PORT ?? 3000);
		Bun.serve({
			port,
			fetch: createHandler({ apiKey: Bun.env.CMD_API_KEY }),
		});
		console.log(`SemIf vs Jev vs Laya 2048: http://localhost:${port}`);
	}
}