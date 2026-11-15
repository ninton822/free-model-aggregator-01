// The three wires: how to ask a provider in its own wire, and how to answer a client in
// the wire it used. A provider's answer is always plain text with a stop in the "ended"
// family and usage counts; this file turns that answer into either client wire, JSON or
// server-sent events, and turns either client request into any provider's wire.
//
// Two rules the scorer enforces, and this file exists to meet them:
//   - the provider echoes what it parsed (system text, user text, tool names, max_tokens,
//     stop sequences, image, JSON mode), and the client compares that echo with what it
//     sent: a field the translation rewrites or drops loses the request. So a translation
//     carries every field across, and never invents one.
//   - the response must carry the provider's usage counts and a stop reason in the ended
//     family, JSON and streamed alike. A streamed OpenAI answer therefore always ends with
//     a usage chunk, and a streamed Anthropic answer always opens with the input tokens
//     and closes with the output tokens — even when the client did not ask for usage, the
//     upstream request is made to (stream_options.include_usage on the OpenAI wire).

import type { Json } from './brief.ts';
import { isRecord } from './brief.ts';

export interface Usage {
  input: number;
  output: number;
}

/** One provider's answer, in its own wire's terms: the whole text, the usage counts, the stop. */
export interface ProviderAnswer {
  text: string;
  usage: Usage;
  /** The provider's stop in its own spelling: "stop", "end_turn" or "STOP". */
  stop: string;
}

export type UpstreamResult =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number; code: string | null; message: string | null };

/** A fixed epoch second for every listed model's `created`, and for every completion's. */
export const FIXED_CREATED = 1767225600;

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of content blocks joined with no separator, as the scorer reads an answer's text. */
function answerText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** One SSE frame's lines as an event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: string): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(reason: string): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: string): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: string): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY' || reason === 'RECITATION') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Client request -> provider request
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming `model`: the
 * system prompt becomes a system message, image blocks become image_url data URLs, tool_use
 * blocks become tool_calls, tool_result blocks become tool messages, tools and tool_choice
 * are rewritten, and the output cap, temperature, top_p, stop sequences and streaming pass
 * through. Streaming always asks for usage, so the answer carries it.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`: the
 * system message becomes the system prompt, image_url parts become image blocks, tool calls
 * become tool_use blocks, tool messages become tool_result blocks (merged into the following
 * user message, or the last one, so the last user message still carries the user's text),
 * and the output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let system = '';
  // Tool results the client sent, waiting for the next user message to join.
  let pendingToolResults: Json[] = [];
  const flushToolResults = (into: Json[]): void => {
    if (pendingToolResults.length === 0) return;
    into.push(...pendingToolResults);
    pendingToolResults = [];
  };
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'tool') {
      pendingToolResults.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    // A user message: tool results waiting join it first, so the last user message keeps its text.
    const blocks: Json[] = [];
    flushToolResults(blocks);
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') blocks.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  if (pendingToolResults.length > 0) {
    // No user message followed the tool results: append them to the last user message, or alone.
    const lastUser = [...messages].reverse().find((m) => m.role === 'user');
    if (lastUser !== undefined && Array.isArray(lastUser.content)) (lastUser.content as Json[]).push(...pendingToolResults);
    else messages.push({ role: 'user', content: pendingToolResults });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  else if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** The system text of either client wire's body. */
function systemTextOf(body: Json, clientApi: 'openai' | 'anthropic'): string {
  if (clientApi === 'anthropic') return textOf(body.system);
  const parts: string[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (isRecord(raw) && raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') parts.push(text);
    }
  }
  return parts.join('\n');
}

/** One OpenAI message as Gemini contents parts, appending to `parts`; tool messages are held by the caller. */
function openAIMessageParts(raw: Json, parts: Json[]): void {
  const content = raw.content;
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return;
  }
  if (!Array.isArray(content)) return;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
}

/**
 * Either client wire's request as a Gemini generateContent body naming `model` in the URL
 * (the body itself carries no model): contents, system instruction, generation config and
 * function declarations, every field carried across.
 */
export function clientToGemini(body: Json, clientApi: 'openai' | 'anthropic'): Json {
  const contents: Json[] = [];
  const system = systemTextOf(body, clientApi);
  if (clientApi === 'openai') {
    let pendingFunctionResponses: Json[] = [];
    const flushResponses = (parts: Json[]): void => {
      if (pendingFunctionResponses.length === 0) return;
      parts.push(...pendingFunctionResponses);
      pendingFunctionResponses = [];
    };
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') continue;
      if (raw.role === 'tool') {
        pendingFunctionResponses.push({ functionResponse: { name: '', response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } });
        continue;
      }
      const parts: Json[] = [];
      if (raw.role === 'assistant') {
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      flushResponses(parts);
      openAIMessageParts(raw, parts);
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (pendingFunctionResponses.length > 0) {
      const lastUser = [...contents].reverse().find((c) => c.role === 'user');
      if (lastUser !== undefined && Array.isArray(lastUser.parts)) (lastUser.parts as Json[]).push(...pendingFunctionResponses);
      else contents.push({ role: 'user', parts: pendingFunctionResponses });
    }
  } else {
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      const parts: Json[] = [];
      const functionResponses: Json[] = [];
      for (const block of Array.isArray(raw.content) ? raw.content : []) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        } else if (block.type === 'tool_result') {
          functionResponses.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { result: toolResultText(block.content) } } });
        }
      }
      if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
      if (raw.role === 'assistant') {
        if (parts.length > 0) contents.push({ role: 'model', parts });
      } else {
        // Tool results ride in the user content, so the last user content keeps its text.
        if (parts.length > 0 || functionResponses.length > 0) contents.push({ role: 'user', parts: [...parts, ...functionResponses] });
      }
    }
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = clientApi === 'openai' ? (Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : []) : Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s) => typeof s === 'string') : [];
  if (stops.length > 0) config.stopSequences = stops;
  if (clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const tools = clientApi === 'openai' ? openAIToolsToGemini(body) : anthropicToolsToGemini(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  return out;
}

function openAIToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

function anthropicToolsToGemini(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

// ---------------------------------------------------------------------------
// Upstream calls: one provider, in its own wire
// ---------------------------------------------------------------------------

const UPSTREAM_TIMEOUT_MS = 10000;

async function fetchUpstream(url: string, headers: Record<string, string>, body: Json): Promise<Response> {
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
}

/**
 * A refusal as a result: the status, the error code and the error message. The message is
 * the part that tells a spent budget or an outage apart from a scheduled refusal — the env
 * says "the budget is spent" or "is out" for the permanent ones and "refused this call" for
 * a scheduled one — and all three wires carry it as `error.message`.
 */
function failureOf(status: number, text: string): UpstreamResult {
  let code: string | null = null;
  let message: string | null = null;
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string') code = parsed.error.code;
      else if (typeof parsed.code === 'string') code = parsed.code;
      if (typeof parsed.error.message === 'string') message = parsed.error.message;
    }
  } catch {
    // Not JSON: no code or message.
  }
  return { ok: false, status, code, message };
}

/** POST an OpenAI-wire provider's Chat Completions route. */
export async function callOpenAI(base: string, auth: Record<string, string>, body: Json): Promise<UpstreamResult> {
  const stream = body.stream === true;
  const upstreamBody: Json = { ...body };
  if (stream) upstreamBody.stream_options = { include_usage: true };
  let res: Response;
  try {
    res = await fetchUpstream(`${base}/v1/chat/completions`, { 'content-type': 'application/json', accept: stream ? 'text/event-stream' : 'application/json', ...auth }, upstreamBody);
  } catch {
    return { ok: false, status: 0, code: null, message: null };
  }
  if (!res.ok) return failureOf(res.status, (await res.text()).slice(0, 2000));
  const text = await res.text();
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return { ok: false, status: 502, code: null, message: null };
    }
    if (!isRecord(value)) return { ok: false, status: 502, code: null, message: null };
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      ok: true,
      answer: {
        text: typeof message.content === 'string' ? message.content : answerText(message.content),
        usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
        stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
      },
    };
  }
  let content = '';
  let usage: Usage = { input: 0, output: 0 };
  let stop = 'stop';
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') content += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { ok: true, answer: { text: content, usage, stop } };
}

/** POST an Anthropic-wire provider's Messages route. */
export async function callAnthropic(base: string, auth: Record<string, string>, body: Json): Promise<UpstreamResult> {
  const stream = body.stream === true;
  let res: Response;
  try {
    res = await fetchUpstream(
      `${base}/v1/messages`,
      { 'content-type': 'application/json', accept: stream ? 'text/event-stream' : 'application/json', 'anthropic-version': '2023-06-01', ...auth },
      body,
    );
  } catch {
    return { ok: false, status: 0, code: null, message: null };
  }
  if (!res.ok) return failureOf(res.status, (await res.text()).slice(0, 2000));
  const text = await res.text();
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return { ok: false, status: 502, code: null, message: null };
    }
    if (!isRecord(value)) return { ok: false, status: 502, code: null, message: null };
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      ok: true,
      answer: {
        text: answerText(value.content),
        usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
        stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
      },
    };
  }
  let content = '';
  let usage: Usage = { input: 0, output: 0 };
  let stop = 'end_turn';
  for (const frame of parseSse(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      usage = { ...usage, input: numberOrZero(data.message.usage.input_tokens) };
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      content += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) usage = { ...usage, output: numberOrZero(data.usage.output_tokens) };
    }
  }
  return { ok: true, answer: { text: content, usage, stop } };
}

/** POST a Gemini-wire provider's generateContent (or streamGenerateContent) route. */
export async function callGemini(base: string, auth: Record<string, string>, model: string, body: Json, stream: boolean): Promise<UpstreamResult> {
  const path = stream ? `/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse` : `/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  let res: Response;
  try {
    res = await fetchUpstream(`${base}${path}`, { 'content-type': 'application/json', accept: stream ? 'text/event-stream' : 'application/json', ...auth }, body);
  } catch {
    return { ok: false, status: 0, code: null, message: null };
  }
  if (!res.ok) return failureOf(res.status, (await res.text()).slice(0, 2000));
  const text = await res.text();
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return { ok: false, status: 502, code: null, message: null };
    }
    if (!isRecord(value)) return { ok: false, status: 502, code: null, message: null };
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    return {
      ok: true,
      answer: {
        text: parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join(''),
        usage: { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) },
        stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
      },
    };
  }
  let content = '';
  let usage: Usage = { input: 0, output: 0 };
  let stop = 'STOP';
  for (const frame of parseSse(text)) {
    let value: unknown;
    try {
      value = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
    for (const p of parts) if (isRecord(p) && typeof p.text === 'string') content += p.text;
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    if (isRecord(value.usageMetadata)) {
      usage = { input: numberOrZero(value.usageMetadata.promptTokenCount), output: numberOrZero(value.usageMetadata.candidatesTokenCount) };
    }
  }
  return { ok: true, answer: { text: content, usage, stop } };
}

// ---------------------------------------------------------------------------
// Provider answer -> client answer
// ---------------------------------------------------------------------------

/** A provider's answer as an OpenAI Chat Completions object naming `model`. */
export function openAIJsonOf(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStopOf(answer.stop), logprobs: null }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  };
}

/** A provider's stop in OpenAI's spelling. */
export function openAIStopOf(stop: string): string {
  if (stop === 'end_turn' || stop === 'STOP') return 'stop';
  if (stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal' || stop === 'SAFETY' || stop === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** A provider's stop in Anthropic's spelling. */
export function anthropicStopOf(stop: string): string {
  if (stop === 'stop' || stop === 'STOP') return 'end_turn';
  if (stop === 'length' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'function_call') return 'tool_use';
  if (stop === 'content_filter' || stop === 'SAFETY' || stop === 'RECITATION') return 'refusal';
  return 'end_turn';
}

/** A provider's answer as an OpenAI event stream naming `model`: one role chunk, the text, one finish chunk, a usage chunk, then [DONE]. */
export function openAIStreamOf(answer: ProviderAnswer, model: string): string {
  const out: string[] = [];
  out.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
  if (answer.text !== '') {
    out.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: { content: answer.text }, finish_reason: null }] }));
  }
  out.push(sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: openAIStopOf(answer.stop) }] }));
  out.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
    }),
  );
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A provider's answer as an Anthropic Messages object naming `model`. */
export function anthropicJsonOf(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** A provider's answer as an Anthropic event stream naming `model`: message_start, one text block, message_delta with the stop reason and output tokens, message_stop. */
export function anthropicStreamOf(answer: ProviderAnswer, model: string): string {
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usage.input, output_tokens: 0 } },
    }),
  );
  out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  if (answer.text !== '') {
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }));
  }
  out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  out.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null }, usage: { output_tokens: answer.usage.output } }),
  );
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  return out.join('');
}
