// One server that answers OpenAI Chat Completions and Anthropic Messages requests with the
// best free model the trial's providers offer, in either wire, JSON or streamed.
//
// The trial: the sandbox sets MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE and MAKEFUN_TRIAL_SEED.
// The brief (the prompt file) holds the provider directory, the sealed quality table and
// the workload manifest. This server probes each provider's catalog route, builds the
// catalog it registers, then serves the twenty requests one at a time and exits when the
// env says the trial is done.
//
// Routing: for each request, the requirements are read from the request itself (a tools
// array, an image part, a JSON output mode, a very long input) and from the manifest entry
// the request's position implies — the env sends the manifest in order, one request at a
// time, so the two agree, and their union is the safe requirement. Every free model that
// meets them is a candidate, ranked by the sealed rating, then provider id, then model id.
// The first candidate that answers serves the request; a 402/429/5xx or a network error
// fails over to the next. A provider that fails three calls in a row is treated as spent
// or out and is not tried again: the failure schedules hold at most two adjacent refusals,
// so three in a row means the budget is gone or the provider is down. Everything is
// derived from the brief, the requests and the providers' answers — no clock, no randomness.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { env } from 'node:process';
import {
  readBrief,
  buildCatalog,
  probeProvider,
  joinBase,
  authHeaders,
  ratingOfModel,
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  type Brief,
  type CatalogReport,
  type ProviderEntry,
  type Requirement,
  type Capabilities,
  type Json,
} from './lib/brief.ts';
import { isRecord } from './lib/brief.ts';
import {
  anthropicToOpenAI,
  openAIToAnthropic,
  clientToGemini,
  callOpenAI,
  callAnthropic,
  callGemini,
  openAIJsonOf,
  openAIStreamOf,
  anthropicJsonOf,
  anthropicStreamOf,
  FIXED_CREATED,
  type UpstreamResult,
} from './lib/wires.ts';

const REQUEST_DEADLINE_MS = 25000;
const POLL_INTERVAL_MS = 100;
const POLL_GUARD_MS = 170000;
const BODY_LIMIT_BYTES = 4 * 1024 * 1024;
/** An input longer than this is a long-context request: the manifest's plain requests are a few hundred characters, its long-context ones over ten thousand. */
const LONG_INPUT_CHARS = 1000;
/** A provider that fails this many calls in a row is spent or out: the schedules hold at most two adjacent refusals. */
const DEAD_AFTER_FAILURES = 3;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

interface ModelEntry {
  provider: ProviderEntry;
  modelId: string;
  rating: number;
  caps: Capabilities;
}

interface ProviderState {
  calls: number;
  failures: number;
  dead: boolean;
}

class Aggregator {
  readonly envUrl: string;
  readonly brief: Brief;
  catalog: CatalogReport = { providers: [] };
  private models: ModelEntry[] = [];
  private states = new Map<string, ProviderState>();
  private requestCount = 0;

  constructor(envUrl: string, brief: Brief) {
    this.envUrl = envUrl;
    this.brief = brief;
  }

  /** Probe every provider's catalog route, in directory order, and build the catalog and the model index. */
  async init(): Promise<void> {
    if (this.envUrl === '') return;
    const probes = [];
    for (const provider of this.brief.directory.providers) {
      const probe = await probeProvider(this.envUrl, provider);
      probes.push(probe);
      if (!probe.ok) process.stderr.write(`free-model-aggregator-01: catalog probe ${provider.id} failed: ${probe.error ?? `HTTP ${probe.status}`}\n`);
    }
    this.catalog = buildCatalog(probes, this.brief.ratings);
    for (const probe of probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingOfModel(this.brief.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        this.models.push({ provider: probe.provider, modelId: model.id, rating: row?.rating_milli ?? 0, caps: capabilitiesOf(model, row) });
      }
    }
    process.stderr.write(
      `free-model-aggregator-01: catalog holds ${this.catalog.providers.length} providers, ${this.models.length} free models` +
        (this.brief.ratings === null ? ' (no quality table in the brief)' : '') +
        '\n',
    );
  }

  private stateOf(providerId: string): ProviderState {
    let state = this.states.get(providerId);
    if (state === undefined) {
      state = { calls: 0, failures: 0, dead: false };
      this.states.set(providerId, state);
    }
    return state;
  }

  /** The requirements a request carries, read from the request itself. */
  private inferredRequirements(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
    const requires: Requirement = {};
    if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
    if (this.bodyHasImage(clientApi, body)) requires.vision = true;
    if (clientApi === 'openai') {
      if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) requires.json = true;
    } else if (/\bjson\b/i.test(this.bodyText(clientApi, body))) {
      // An Anthropic request asks for JSON in its words; the wire has no JSON mode of its own.
      requires.json = true;
    }
    if (this.bodyText(clientApi, body).length > LONG_INPUT_CHARS) requires.min_context = 200000;
    return requires;
  }

  /** Whether any user message carries an image, in either client wire. */
  private bodyHasImage(clientApi: 'openai' | 'anthropic', body: Json): boolean {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of messages) {
      if (!isRecord(raw)) continue;
      const content = raw.content;
      if (!Array.isArray(content)) continue;
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (clientApi === 'openai' && part.type === 'image_url') return true;
        if (clientApi === 'anthropic' && part.type === 'image') return true;
      }
    }
    return false;
  }

  /** The system and user text of a request, in either client wire, as one string. */
  private bodyText(clientApi: 'openai' | 'anthropic', body: Json): string {
    const chunks: string[] = [];
    if (clientApi === 'anthropic' && typeof body.system === 'string') chunks.push(body.system);
    if (clientApi === 'anthropic' && Array.isArray(body.system)) {
      for (const block of body.system) if (isRecord(block) && typeof block.text === 'string') chunks.push(block.text);
    }
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      const content = raw.content;
      if (raw.role === 'system' && typeof content === 'string') chunks.push(content);
      else if (typeof content === 'string') chunks.push(content);
      else if (Array.isArray(content)) {
        for (const part of content) if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') chunks.push(part.text);
      }
    }
    return chunks.join('\n');
  }

  private unionRequirements(a: Requirement, b: Requirement): Requirement {
    const out: Requirement = {};
    if (a.tools === true || b.tools === true) out.tools = true;
    if (a.vision === true || b.vision === true) out.vision = true;
    if (a.json === true || b.json === true) out.json = true;
    const min = Math.max(a.min_context ?? 0, b.min_context ?? 0);
    if (min > 0) out.min_context = min;
    return out;
  }

  /**
   * The candidates for one request: every free model meeting its requirements, best rating
   * first, then provider id, then model id. An OpenAI client asking for JSON mode never gets
   * an Anthropic-wire provider: that wire can never echo JSON mode, so the request would fail
   * its JSON check outright.
   */
  private candidates(requires: Requirement, clientApi: 'openai' | 'anthropic', body: Json): ModelEntry[] {
    const jsonMode = clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
    return this.models
      .filter((m) => meetsRequirements(requires, m.caps))
      .filter((m) => !(jsonMode && m.provider.wire === 'anthropic'))
      .sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0));
  }

  private async callProvider(entry: ModelEntry, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
    const base = joinBase(this.envUrl, entry.provider.base);
    const auth = authHeaders(entry.provider.auth, entry.provider.key);
    if (entry.provider.wire === 'openai') {
      const upstream: Json = clientApi === 'openai' ? { ...body, model: entry.modelId, stream } : anthropicToOpenAI(body, entry.modelId);
      return callOpenAI(base, auth, upstream);
    }
    if (entry.provider.wire === 'anthropic') {
      const upstream: Json = clientApi === 'anthropic' ? { ...body, model: entry.modelId } : openAIToAnthropic(body, entry.modelId);
      return callAnthropic(base, auth, upstream);
    }
    return callGemini(base, auth, entry.modelId, clientToGemini(body, clientApi), stream);
  }

  /**
   * Serve one client request: pick the best free model that meets its requirements, fail
   * over down the ranking until one answers, and return the answer in the client's wire.
   * Returns null when no provider answered.
   */
  async serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ modelId: string; stream: boolean; json: Json; sse: string } | null> {
    const index = this.requestCount;
    this.requestCount += 1;
    const manifestEntry = this.brief.workload?.manifest[index] ?? null;
    const inferred = this.inferredRequirements(clientApi, body);
    const requires = manifestEntry === null ? inferred : this.unionRequirements(inferred, manifestEntry.requires);
    if (manifestEntry !== null && (manifestEntry.api !== clientApi || manifestEntry.stream !== (body.stream === true))) {
      process.stderr.write(`free-model-aggregator-01: request ${index + 1} (${manifestEntry.id}) does not match its manifest entry (api/stream differ); serving by the body's own shape\n`);
    }
    const stream = body.stream === true;
    const list = this.candidates(requires, clientApi, body);
    const deadline = Date.now() + REQUEST_DEADLINE_MS;
    for (const entry of list) {
      if (Date.now() > deadline) break;
      const state = this.stateOf(entry.provider.id);
      if (state.dead) continue;
      state.calls += 1;
      let result: UpstreamResult;
      try {
        result = await this.callProvider(entry, clientApi, body, stream);
      } catch {
        result = { ok: false, status: 0, code: null, message: null };
      }
      if (result.ok) {
        state.failures = 0;
        process.stderr.write(`free-model-aggregator-01: request ${index + 1} served by ${entry.provider.id}/${entry.modelId} (rating ${entry.rating})\n`);
        if (clientApi === 'openai') {
          return { modelId: entry.modelId, stream, json: openAIJsonOf(result.answer, entry.modelId), sse: openAIStreamOf(result.answer, entry.modelId) };
        }
        return { modelId: entry.modelId, stream, json: anthropicJsonOf(result.answer, entry.modelId), sse: anthropicStreamOf(result.answer, entry.modelId) };
      }
      // A refusal that says the budget is spent or the provider is out is permanent for this
      // trial: the provider can never serve again, so stop trying it at once instead of
      // spending further attempts on it before the consecutive-failure rule would. A scheduled
      // refusal only says "refused this call" and leaves the provider usable.
      const message = result.message ?? '';
      if (message.includes('budget is spent') || message.includes(' is out')) {
        state.dead = true;
        process.stderr.write(`free-model-aggregator-01: provider ${entry.provider.id} is done for this trial (${message}); not trying it again\n`);
      } else {
        state.failures += 1;
        if (state.failures >= DEAD_AFTER_FAILURES) {
          state.dead = true;
          process.stderr.write(`free-model-aggregator-01: provider ${entry.provider.id} failed ${state.failures} calls in a row (last HTTP ${result.status}); treating it as spent or out\n`);
        } else {
          process.stderr.write(`free-model-aggregator-01: ${entry.provider.id}/${entry.modelId} refused (HTTP ${result.status}${result.code === null ? '' : `, ${result.code}`}); failing over\n`);
        }
      }
    }
    process.stderr.write(`free-model-aggregator-01: request ${index + 1} failed: no provider answered\n`);
    return null;
  }

  /** The free models, in both list shapes, for GET /v1/models. */
  modelLists(): { openai: Json; anthropic: Json } {
    const listed: { id: string; provider: string }[] = [];
    for (const provider of this.catalog.providers) {
      for (const model of provider.models) if (model.free) listed.push({ id: model.id, provider: provider.id });
    }
    listed.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : a.provider < b.provider ? -1 : a.provider > b.provider ? 1 : 0));
    return {
      openai: {
        object: 'list',
        data: listed.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
      },
      anthropic: {
        data: listed.map((m) => ({ id: m.id, type: 'model', display_name: m.id, created_at: '2026-01-01T00:00:00Z' })),
        has_more: false,
        first_id: listed.length > 0 ? listed[0].id : null,
        last_id: listed.length > 0 ? listed[listed.length - 1].id : null,
      },
    };
  }
}

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    size += buffer.length;
    if (size > BODY_LIMIT_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, value: Json): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(value));
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

async function register(envUrl: string, port: number, catalog: CatalogReport): Promise<boolean> {
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const data: unknown = await res.json();
        if (isRecord(data) && data.ok === true) {
          process.stderr.write(`free-model-aggregator-01: registered on port ${port}; the env expects ${typeof data.requests === 'number' ? data.requests : '?'} requests\n`);
          return true;
        }
      }
      process.stderr.write(`free-model-aggregator-01: register attempt ${attempt + 1} answered HTTP ${res.status}\n`);
    } catch (err) {
      process.stderr.write(`free-model-aggregator-01: register attempt ${attempt + 1} failed: ${(err as Error).message}\n`);
    }
    await sleep(200);
  }
  return false;
}

async function waitForDone(envUrl: string): Promise<void> {
  const started = Date.now();
  for (;;) {
    try {
      const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const data: unknown = await res.json();
        if (isRecord(data) && data.done === true) return;
      }
    } catch {
      // The env may be briefly unreachable; keep polling.
    }
    if (Date.now() - started > POLL_GUARD_MS) {
      process.stderr.write('free-model-aggregator-01: the trial did not finish within the guard window; stopping\n');
      return;
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listMode = args.includes('--list');
  const envUrl = (env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const promptFile = env.MAKEFUN_PROMPT_FILE ?? '';
  const brief = readBrief(promptFile);
  if (brief.error !== null && promptFile !== '') process.stderr.write(`free-model-aggregator-01: ${brief.error}\n`);

  const aggregator = new Aggregator(envUrl, brief);
  await aggregator.init();

  if (listMode) {
    process.stdout.write(`${JSON.stringify(envUrl === '' ? { providers: [] } : aggregator.catalog)}\n`);
    return;
  }

  const server = createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? '/', 'http://127.0.0.1');
      if (req.method === 'GET' && url.pathname === '/v1/models') {
        const lists = aggregator.modelLists();
        sendJson(res, 200, req.headers['x-api-key'] !== undefined ? lists.anthropic : lists.openai);
        return;
      }
      if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
        const clientApi = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
        const body = await readJsonBody(req);
        if (body === null) {
          sendJson(
            res,
            400,
            clientApi === 'openai'
              ? { error: { message: 'the request body is not a JSON object', type: 'invalid_request_error', code: null } }
              : { type: 'error', error: { type: 'invalid_request_error', message: 'the request body is not a JSON object' } },
          );
          return;
        }
        const served = await aggregator.serve(clientApi, body);
        if (served === null) {
          sendJson(
            res,
            503,
            clientApi === 'openai'
              ? { error: { message: 'no provider answered this request', type: 'server_error', code: null } }
              : { type: 'error', error: { type: 'overloaded_error', message: 'no provider answered this request' } },
          );
          return;
        }
        if (served.stream) sendSse(res, served.sse);
        else sendJson(res, 200, served.json);
        return;
      }
      sendJson(res, 404, { error: { message: `no such route: ${url.pathname}`, type: 'invalid_request_error', code: null } });
    })().catch((err: unknown) => {
      process.stderr.write(`free-model-aggregator-01: request handler failed: ${(err as Error).message}\n`);
      if (!res.headersSent) sendJson(res, 500, { error: { message: 'internal error', type: 'server_error', code: null } });
      else res.end();
    });
  });

  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  process.stderr.write(`free-model-aggregator-01: listening on port ${port}\n`);
  if (envUrl !== '') {
    await register(envUrl, port, aggregator.catalog);
    await waitForDone(envUrl);
  } else {
    process.stderr.write('free-model-aggregator-01: no MAKEFUN_ENV_URL set; serving without a trial\n');
  }
  server.close();
  process.exit(0);
}

main().catch((err: unknown) => {
  process.stderr.write(`free-model-aggregator-01: fatal: ${(err as Error).message}\n`);
  process.exit(1);
});
