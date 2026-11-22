/**
 * Generic pi-ai-backed implementation of the Harness LLM seam.
 *
 * Each configuration resolution produces one snapshot — the profiles plus a
 * `Models` collection holding the `Provider` each route built — and an
 * operation captures a whole snapshot before its first `await`. Provider
 * listings update only the snapshot's catalog view. A configuration change
 * builds a *new* collection rather than mutating the one in use, because
 * `Models.streamSimple()` is lazy: it resolves the provider
 * when the stream is first consumed, which is after the credential await, so a
 * mutated collection would let a request that started under one configuration
 * finish under another — or fail with a provider that no longer exists. This is
 * what makes the seam's per-step call freeze (`llm.prepareCall()`) hold all the
 * way down: switching models mid-reply takes effect on the next step, never
 * inside the one in flight.
 *
 * A route naming a credential reference still resolves it through the harness
 * seam and passes it as the request's `apiKey` option, which pi-ai treats as
 * the highest-priority auth override — that is what keeps the fail-loud
 * reference semantics. Everything that override does not cover reaches pi-ai
 * through the collection's own auth: the credential store holds the records a
 * login wrote and a refresh rotates, and the auth context answers the ambient
 * questions a provider asks while resolving. Both are stable across snapshots,
 * so a configuration change rebuilds the collection without forgetting who is
 * signed in.
 *
 * @module dsh-llm-pi-ai/adapter
 */

import { createHash } from 'node:crypto'
import type {
  Api,
  AuthContext,
  CredentialStore,
  Model,
  Models,
  ModelThinkingLevel,
  MutableModels,
  SimpleStreamOptions,
  ThinkingLevel,
} from '@earendil-works/pi-ai'
import {
  attributionHeaders,
  contentHasImage,
  LlmAdapter,
  LlmError,
} from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  ImageAttachmentAccess,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  PreparedAdapterCall,
  ReasoningEffortId as ReasoningEffortIdType,
  ResolvedRetryPolicy,
  StreamChunk,
  ToolSchema,
} from '@deepseek-ai/dsh-llm'
import type { AttachmentStore, ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import { brandString } from '@deepseek-ai/dsh-brand'
import { deadline, idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout'
import { deepEqualJson } from '@deepseek-ai/dsh-util-values'
import type { ResolvedPiAiProviderProfile } from './config.ts'
import { requestImageTarget, toPiContext } from './context.ts'
import { resolveImageRequestBudget } from './image-policy.ts'
import { conversationUpdates, createModels, getSupportedThinkingLevels } from './models.ts'
import { toStreamChunks } from './stream.ts'
import { bindThinkingToPrefix } from './prefix-binding.ts'
import type { PrefixDigests } from './prefix-binding.ts'
import type { ProviderListingAuth } from './sync.ts'
import type { PiAiRequestHeaderContext } from './types.ts'

/**
 * Initially active declaration added to an Anthropic request whose projected
 * tools are all deferred, because Anthropic rejects a tool list without one.
 */
const ANTHROPIC_TOOL_UPDATE_PLACEHOLDER: ToolSchema = {
  name: 'DeferredToolPlaceholder',
  description: 'Reserved placeholder that keeps deferred tool loading active; never call this tool.',
  parameters: { type: 'object', properties: {} },
}

interface ProviderCatalogState {
  /** Credential and endpoint identity currently visible to readers. */
  activeScope?: string
  /** Last successful catalog in the active scope. */
  synced?: { scope: string; models: readonly Model<Api>[] }
  /** Publication version, incremented whenever scope changes. */
  generation: number
  /** Shared work for one scope; retained until the operation actually settles. */
  inflight?: { scope: string; operation: Promise<void> }
}

/** One resolution's frozen view: the profiles and the collection built from them. */
interface PiAiSnapshot {
  /** The resolved profiles this collection was built from, used as its identity. */
  profiles: ReadonlyMap<string, ResolvedPiAiProviderProfile>
  /** Providers for exactly those profiles; request routing stays on this collection. */
  models: Models
  /** Credential-scoped catalog state by provider. */
  catalogs: Map<string, ProviderCatalogState>
}

/** Hash credentials rather than retaining a second readable copy as a catalog key. */
function listingScope(auth: ProviderListingAuth): string {
  const headers = auth.headers
  return createHash('sha256').update(JSON.stringify({
    apiKey: auth.apiKey,
    headers,
    baseURL: auth.baseURL,
  })).digest('hex')
}

/** Wait for shared listing work without making one caller's cancellation cancel the others. */
async function waitForSync(task: Promise<void>, signal: AbortSignal | undefined): Promise<void> {
  if (signal === undefined) return task
  signal.throwIfAborted()
  const abandoned = Promise.withResolvers<never>()
  const abandon = (): void => { abandoned.reject(signal.reason) }
  signal.addEventListener('abort', abandon, { once: true })
  try {
    await Promise.race([task, abandoned.promise])
  } finally {
    signal.removeEventListener('abort', abandon)
  }
}

/** Constructor options for {@link PiAiAdapter}: profile, credential, catalog, and request hooks the plugin owns. */
export interface PiAiAdapterOptions {
  /** Current validated profiles by provider route; called once per operation. */
  profiles: () => ReadonlyMap<string, ResolvedPiAiProviderProfile>
  /**
   * Resolve the credential for one already-resolved profile; called once per
   * stream call and frozen for that call, and by `listModels`, which offers no
   * models while it throws `MISSING_CREDENTIAL`. `undefined` defers to the route's own
   * pi-ai auth, which for an installed catalog route is its provider-native
   * ambient discovery; the plugin allows that only for a profile naming no
   * credential at all, because a named reference that misses throws `LlmError`
   * `MISSING_CREDENTIAL` rather than falling back.
   */
  resolveApiKey: (provider: string, profile: ResolvedPiAiProviderProfile) => Promise<string | undefined>
  /**
   * Load one route's latest provider catalog. Omission keeps only its installed or explicitly configured models.
   * The result replaces the route's advertised baseline after a successful load; failures retain the last good catalog.
   */
  syncModels?: (
    provider: string,
    profile: ResolvedPiAiProviderProfile,
    auth: ProviderListingAuth,
    signal: AbortSignal,
  ) => Promise<readonly Model<Api>[] | undefined>
  /** Publish a successfully changed provider catalog after readers can resolve it. */
  onModelsUpdated?: (provider: string) => void
  /** Report a failed automatic catalog load while the previous catalog remains available. */
  onModelSyncError?: (detail: { provider: string; error: unknown }) => void
  /**
   * How every collection this adapter builds resolves auth the request-level
   * `apiKey` override does not cover. Required rather than optional: a
   * collection built without them gets pi-ai's in-memory default store, which
   * is empty at every boot and discarded on every configuration change, so a
   * route whose only method is a login would report itself unconfigured on
   * every request no matter how often the human signed in.
   */
  auth: PiAiAuthInjection
  /**
   * Synchronously prepare this call's headers before SDK dispatch. The input
   * contains a fresh copy of profile headers; attribution is enforced afterward.
   * Omission preserves profile headers. Throws fail the call with REQUEST_HEADERS
   * before inference I/O. Do not retain or mutate headers after returning.
   * Non-TypeError messages become public diagnostics and must exclude secrets;
   * TypeError details and non-Error values are replaced with safe guidance.
   * @param request - immutable routing metadata from the prepared call.
   * @param headers - request-owned headers initialized from the profile.
   * @returns headers to pass to pi-ai after applying Harness attribution.
   */
  prepareHeaders?: (request: PiAiRequestHeaderContext, headers: Headers) => Headers
  /** Resolve the optional durable attachment service at request time. */
  resolveAttachments?: () => AttachmentStore | undefined
  /** Bridge one attachment reference into the current model-tool execution world. */
  resolveImageAccess?: (attachments: AttachmentStore, ref: ImageAttachmentRef) => ImageAttachmentAccess | undefined
  /**
   * Observe one assistant history message degrading to provider-neutral
   * conversion because its stored replay state is unusable by this build.
   */
  onReplayDegrade?: (detail: { provider: string; model: string; reason: string }) => void
  /**
   * Observe one request that sent replayed assistant messages without their
   * signed Anthropic thinking because the request prefix changed or was never
   * recorded. Stale messages stay omitted, so later requests of the
   * conversation report them again.
   */
  onThinkingOmitted?: (detail: { provider: string; model: string; count: number }) => void
}

/** The two auth injectables a pi-ai collection is built with. */
export interface PiAiAuthInjection {
  /** Durable storage for credentials pi-ai itself writes: logins, and the refreshes it runs under its own lock. */
  credentials: CredentialStore
  /** Ambient lookups a provider performs while resolving its own auth. */
  authContext: AuthContext
}

/** Copy profile stream knobs into pi-ai's common option vocabulary. */
function profileOptions(
  profile: ResolvedPiAiProviderProfile,
  reasoning: ModelThinkingLevel | undefined,
  apiKey: string | undefined,
): SimpleStreamOptions {
  const enabledReasoning: ThinkingLevel | undefined = reasoning === 'off' ? undefined : reasoning
  return {
    ...apiKey === undefined ? {} : { apiKey },
    ...enabledReasoning === undefined ? {} : { reasoning: enabledReasoning },
    ...profile.thinkingBudgets === undefined ? {} : { thinkingBudgets: profile.thinkingBudgets },
    ...profile.cacheRetention === undefined ? {} : { cacheRetention: profile.cacheRetention },
    ...profile.transport === undefined ? {} : { transport: profile.transport },
    ...profile.timeoutMs === undefined ? {} : { timeoutMs: profile.timeoutMs },
    ...profile.websocketConnectTimeoutMs === undefined ? {} : { websocketConnectTimeoutMs: profile.websocketConnectTimeoutMs },
    // The agent recovery layer owns visible attempts; one adapter call is one SDK attempt.
    maxRetries: 0,
  }
}

/**
 * The profile default this exact model can actually take, for DESCRIBING it.
 * A configured level the model does not support yields none rather than
 * throwing: `resolveModel` builds the model catalog, and a catalog that fails
 * takes its whole provider out of every picker — so one mis-set profile field
 * would hide every model on the route, including the ones that support the
 * level. The request path still refuses, which is where a bad configuration
 * belongs: describing what a model can do must not fail because a deployment
 * asked it for something it cannot.
 * @param model - the resolved model descriptor.
 * @param effort - the profile's configured level, if any.
 * @returns the level when this model supports it, otherwise undefined.
 */
function describableReasoningLevel(
  model: Model<Api>,
  effort: ReasoningEffortIdType | ModelThinkingLevel | undefined,
): ModelThinkingLevel | undefined {
  if (effort === undefined) return undefined
  return getSupportedThinkingLevels(model).some(level => level === effort)
    ? effort as ModelThinkingLevel
    : undefined
}

/** Validate an explicit Harness/profile effort without invoking pi-ai's clamp. */
function resolveReasoningLevel(
  model: Model<Api>,
  effort: ReasoningEffortIdType | ModelThinkingLevel | undefined,
): ModelThinkingLevel | undefined {
  if (effort === undefined) return undefined
  const supported = getSupportedThinkingLevels(model)
  if (supported.some(level => level === effort)) return effort as ModelThinkingLevel
  throw new LlmError(
    `pi-ai provider "${model.provider}" model "${model.id}" does not support reasoning effort "${effort}"`,
    'UNSUPPORTED_REASONING_EFFORT',
  )
}

/**
 * Selectable reasoning efforts for one model, or nothing at all.
 *
 * A model that carries no reasoning metadata — every hand-declared one, and
 * every catalog model pi-ai marks as non-reasoning — is reported by pi-ai as
 * supporting the single level `off`. Passing that through would offer a control
 * that cannot do what it says: `off` is translated to *omitting* the reasoning
 * option, which for such a model is byte-for-byte the same request as naming no
 * effort — so a provider whose own default is to think would keep thinking with
 * `off` selected. Omitting `reasoning` entirely is the seam's way of saying the
 * capability is unavailable, which leaves the surface offering only the
 * provider's default.
 * @param model - the resolved model descriptor.
 * @param defaultLevel - the profile's configured effort, already validated.
 * @returns the `reasoning` field, or an empty object when none can be offered.
 */
function reasoningInfo(
  model: Model<Api>,
  defaultLevel: ModelThinkingLevel | undefined,
): Pick<LlmResolvedModelInfo, 'reasoning'> | Record<string, never> {
  if (!model.reasoning) return {}
  const levels = getSupportedThinkingLevels(model)
  return {
    reasoning: {
      efforts: levels.map(level => ({
        id: brandString<ReasoningEffortIdType>(level),
        name: `${level.charAt(0).toUpperCase()}${level.slice(1)}`,
      })),
      ...defaultLevel === undefined ? {} : { defaultEffort: brandString<ReasoningEffortIdType>(defaultLevel) },
    },
  }
}

/** Native header TypeErrors can echo credentials; other plugin errors own their public message. */
function requestHeaderDiagnostic(error: unknown): string {
  if (error instanceof TypeError) {
    return 'TypeError during header preparation; check the contributing plugin\'s header names, values, and configuration'
  }
  return error instanceof Error
    ? error.message
    : 'the contributing plugin threw a non-Error value; throw an Error with a safe diagnostic'
}

/** Prepare request-owned headers without allowing contributors to replace attribution. */
function requestHeaders(
  configured: Readonly<Record<string, string>> | undefined,
  options: GenerateOptions<string>,
  baseURL: string,
  prepare: PiAiAdapterOptions['prepareHeaders'],
): Record<string, string> {
  try {
    const initial = new Headers()
    // Profile aliases overwrite case-insensitively; the Headers constructor appends them.
    for (const [name, value] of Object.entries(configured ?? {})) initial.set(name, value)
    const request: PiAiRequestHeaderContext = Object.freeze({
      provider: options.provider,
      model: options.model,
      baseURL,
      ...options.sessionId === undefined ? {} : { sessionId: options.sessionId },
      ...options.purpose === undefined ? {} : { purpose: options.purpose },
    })
    const headers = prepare === undefined ? initial : prepare(request, initial)
    for (const [name, value] of Object.entries(attributionHeaders())) headers.set(name, value)
    return Object.fromEntries(headers)
  } catch (error) {
    throw new LlmError(
      `pi-ai request header preparation failed for provider "${options.provider}": ${requestHeaderDiagnostic(error)}`,
      'REQUEST_HEADERS', { cause: error },
    )
  }
}

/**
 * pi-ai-backed multi-provider adapter. Each operation reads the current
 * profiles, so a configuration change reaches the next request without a
 * restart; model descriptors come from the collection those profiles built.
 */
export class PiAiAdapter extends LlmAdapter {
  private snapshot: PiAiSnapshot | undefined
  private closed = false
  private readonly syncControllers = new Set<AbortController>()
  private readonly syncOperations = new Set<Promise<void>>()

  constructor(private readonly config: PiAiAdapterOptions) {
    super()
  }

  override imageRequestTarget(provider: string, model: string, image: ImageAttachmentRef) {
    const snapshot = this.current()
    const selected = this.modelOf(snapshot, provider, model)
    const profile = this.profileOf(snapshot, provider)
    return requestImageTarget(image, resolveImageRequestBudget(selected, profile))
  }

  /**
   * The snapshot for the current profiles. Resolution memoizes its result, so
   * an unchanged configuration is recognized by identity; a changed one gets a
   * brand-new collection, leaving any snapshot an operation already captured
   * untouched for as long as that operation holds it.
   */
  private current(): PiAiSnapshot {
    const profiles = this.config.profiles()
    if (this.snapshot?.profiles === profiles) return this.snapshot
    const models: MutableModels = createModels(this.config.auth)
    const snapshot: PiAiSnapshot = { profiles, models, catalogs: new Map() }
    for (const profile of profiles.values()) {
      const baseline = profile.piProvider
      if (baseline === undefined) continue
      const state = this.catalogState(snapshot, profile.provider)
      models.setProvider({
        ...baseline,
        getModels: () => {
          const synced = state.synced
          return synced !== undefined && synced.scope === state.activeScope ? synced.models : baseline.getModels()
        },
      })
    }
    this.snapshot = snapshot
    return snapshot
  }

  /** Get or create the credential-scoped catalog state owned by one snapshot. */
  private catalogState(snapshot: PiAiSnapshot, provider: string): ProviderCatalogState {
    let state = snapshot.catalogs.get(provider)
    if (state === undefined) {
      state = { generation: 0 }
      snapshot.catalogs.set(provider, state)
    }
    return state
  }

  /** The profile for one route within one snapshot, or the not-owned failure. */
  private profileOf(snapshot: PiAiSnapshot, provider: string): ResolvedPiAiProviderProfile {
    const profile = snapshot.profiles.get(provider)
    if (profile === undefined) {
      throw new LlmError(`pi-ai adapter does not own provider "${provider}"`, 'NO_ADAPTER')
    }
    return profile
  }

  /** Resolve explicit credentials without falling back, or provider-native auth when no reference is named. */
  private async listingAuth(
    snapshot: PiAiSnapshot,
    provider: string,
    profile: ResolvedPiAiProviderProfile,
    signal?: AbortSignal,
  ): Promise<ProviderListingAuth> {
    const apiKey = await this.config.resolveApiKey(provider, profile)
    if (profile.apiKeyEnv !== undefined) return apiKey === undefined ? {} : { apiKey }
    const resolved = await snapshot.models.getAuth(provider, {
      ...signal === undefined ? {} : { signal },
    })
    const key = apiKey ?? resolved?.auth.apiKey
    const headers = Object.fromEntries(
      Object.entries(resolved?.auth.headers ?? {}).flatMap(([name, value]) =>
        typeof value === 'string' ? [[name, value] as const] : []),
    )
    return {
      ...key === undefined ? {} : { apiKey: key },
      ...Object.keys(headers).length === 0 ? {} : { headers },
      ...resolved?.auth.baseUrl === undefined ? {} : { baseURL: resolved.auth.baseUrl },
    }
  }

  /** Whether this configuration generation can refresh a route from its provider. */
  private canSynchronize(snapshot: PiAiSnapshot, provider: string): boolean {
    const profile = this.profileOf(snapshot, provider)
    return this.config.syncModels !== undefined && profile.followsProviderModels && profile.piProvider !== undefined
  }

  /**
   * Stop admitting listing work, abort outstanding requests, and await their settlement.
   * Late results neither publish catalogs nor emit notifications.
   */
  async dispose(): Promise<void> {
    if (this.closed) return
    this.closed = true
    for (const controller of this.syncControllers) controller.abort('pi-ai adapter disposed')
    await Promise.allSettled([...this.syncOperations])
  }

  /** Refresh one credential scope at a time, retaining only that scope's last successful catalog. */
  private async synchronize(
    snapshot: PiAiSnapshot,
    provider: string,
    auth: ProviderListingAuth,
    signal?: AbortSignal,
  ): Promise<void> {
    if (this.closed) return
    const profile = this.profileOf(snapshot, provider)
    const load = this.config.syncModels
    if (load === undefined || !profile.followsProviderModels || profile.piProvider === undefined) return
    const baseline = profile.piProvider
    const state = this.catalogState(snapshot, provider)
    const scope = listingScope(auth)
    if (state.activeScope !== scope) {
      state.activeScope = scope
      delete state.synced
      state.generation += 1
    }
    const existing = state.inflight
    if (existing?.scope === scope) return waitForSync(existing.operation, signal)
    const generation = state.generation
    const request = new AbortController()
    const operation = (async () => {
      using limit = deadline(request.signal, profile.modelsTimeoutMs, 'PI_AI_MODEL_SYNC_TIMEOUT')
      try {
        const synced = await load(provider, profile, auth, limit.signal)
        if (synced === undefined || this.closed || this.snapshot !== snapshot
          || state.generation !== generation || state.activeScope !== scope) return
        const current = state.synced
        const previous = current !== undefined && current.scope === scope ? current.models : baseline.getModels()
        if (deepEqualJson(previous, synced)) return
        state.synced = { scope, models: synced }
        this.config.onModelsUpdated?.(provider)
      } catch (error: unknown) {
        if (!this.closed && this.snapshot === snapshot && state.generation === generation) {
          this.config.onModelSyncError?.({ provider, error })
        }
      }
    })()
    state.inflight = { scope, operation }
    this.syncControllers.add(request)
    this.syncOperations.add(operation)
    const cleanup = (): void => {
      if (state.inflight?.operation === operation) delete state.inflight
      this.syncControllers.delete(request)
      this.syncOperations.delete(operation)
    }
    void operation.then(cleanup, cleanup)
    await waitForSync(operation, signal)
  }

  /** The configured descriptor for one exact route/model pair within one snapshot. */
  private modelOf(snapshot: PiAiSnapshot, provider: string, model: string): Model<Api> {
    const profile = this.profileOf(snapshot, provider)
    const failure = profile.modelErrors.get(model)
      ?? (profile.piProvider === undefined ? profile.catalogError : undefined)
    if (failure !== undefined) throw new LlmError(failure, 'INVALID_CONFIG')
    const resolved = snapshot.models.getModel(provider, model)
    if (resolved === undefined) {
      throw new LlmError(`pi-ai provider "${provider}" has no configured model "${model}"`, 'UNKNOWN_MODEL')
    }
    return resolved
  }

  override providerInfo(provider: string): LlmProviderInfo {
    // The configured name, not the route key: `displayName` exists so a
    // deployment can label a route, and a label only the configuration surface
    // reads would leave every selector showing the raw key.
    return { id: provider, name: this.current().profiles.get(provider)?.displayName ?? provider }
  }

  override providerRetryPolicy(provider: string): ResolvedRetryPolicy | undefined {
    return this.current().profiles.get(provider)?.retryPolicy
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const snapshot = this.current()
    const profile = this.profileOf(snapshot, provider)
    let auth: ProviderListingAuth
    try {
      auth = await this.listingAuth(snapshot, provider, profile)
    } catch (error) {
      // A route whose named key does not resolve cannot serve a request, so it offers no models until the key is stored.
      if (error instanceof LlmError && error.code === 'MISSING_CREDENTIAL') return []
      throw error
    }
    await this.synchronize(snapshot, provider, auth)
    return snapshot.models.getModels(provider).map(model => ({
      provider,
      id: model.id,
      name: model.name,
      inputModalities: [...model.input],
    }))
  }

  override async resolveModel(
    provider: string,
    model: string,
    signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    const snapshot = this.current()
    return this.modelInfoWithSync(snapshot, provider, model, signal)
  }

  /** Resolve a model, giving an unknown id one chance to arrive through the provider's latest listing. */
  private async modelInfoWithSync(
    snapshot: PiAiSnapshot,
    provider: string,
    model: string,
    signal?: AbortSignal,
  ): Promise<LlmResolvedModelInfo> {
    try {
      return this.modelInfo(snapshot, provider, model)
    } catch (error: unknown) {
      if (!(error instanceof LlmError) || error.code !== 'UNKNOWN_MODEL' || !this.canSynchronize(snapshot, provider)) throw error
    }
    const profile = this.profileOf(snapshot, provider)
    const auth = await this.listingAuth(snapshot, provider, profile, signal)
    await this.synchronize(snapshot, provider, auth, signal)
    return this.modelInfo(snapshot, provider, model)
  }

  private modelInfo(snapshot: PiAiSnapshot, provider: string, model: string): LlmResolvedModelInfo {
    const profile = this.profileOf(snapshot, provider)
    const resolvedModel = this.modelOf(snapshot, provider, model)
    const defaultLevel = describableReasoningLevel(resolvedModel, profile.reasoning)
    // Only a cap the deployment configured is a request default; the
    // catalog's `maxTokens` sizes the model and stops there.
    const configuredMaxTokens = profile.configuredMaxTokens.get(model)
    return {
      provider,
      id: model,
      name: resolvedModel.name,
      inputModalities: [...resolvedModel.input],
      context: { contextWindow: resolvedModel.contextWindow },
      ...configuredMaxTokens === undefined ? {} : { defaultMaxTokens: configuredMaxTokens },
      ...reasoningInfo(resolvedModel, defaultLevel),
      ...conversationUpdates(resolvedModel),
    }
  }

  override async prepareCall(provider: string, model: string, signal?: AbortSignal): Promise<PreparedAdapterCall> {
    const snapshot = this.current()
    return {
      model: await this.modelInfoWithSync(snapshot, provider, model, signal),
      stream: options => this.streamWithSnapshot(options, snapshot),
    }
  }

  stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    return this.streamWithSnapshot(options, this.current())
  }

  private async * streamWithSnapshot(
    options: GenerateOptions,
    snapshot: PiAiSnapshot,
  ): AsyncIterable<StreamChunk> {
    if (options.stop !== undefined) {
      throw new LlmError('llm-pi-ai does not support GenerateOptions.stop', 'UNSUPPORTED_OPTION')
    }
    // One capture per stream call, taken before any await: the profile, the
    // model descriptor, and the collection all come from the same immutable
    // snapshot, and the credential freezes with them. A configuration change
    // mid-request builds a separate snapshot, so this request finishes under
    // the one it started with and the next call picks up the new one.
    const profile = this.profileOf(snapshot, options.provider)
    const model = this.modelOf(snapshot, options.provider, options.model)
    const updates = conversationUpdates(model)
    // pi-ai collapses Anthropic tool updates without an initially active declaration.
    const tools = options.tools ?? []
    const needsPlaceholder = model.api === 'anthropic-messages' && updates.toolUpdate === 'in-history'
      && tools.every(tool => tool.deferLoading === true)
      && options.messages.some(message => message.role === 'developer'
        && message.content.some(block => block.type === 'tool-addition' || block.type === 'tool-removal'))
    const request = needsPlaceholder
      ? { ...options, tools: [ANTHROPIC_TOOL_UPDATE_PLACEHOLDER, ...tools] }
      : options
    const reasoning = resolveReasoningLevel(
      model,
      options.reasoningEffort ?? profile.reasoning,
    )
    const apiKey = await this.config.resolveApiKey(options.provider, profile)

    const consumer = new AbortController()
    const upstream = options.signal === undefined
      ? consumer.signal
      : AbortSignal.any([options.signal, consumer.signal])
    const streamIdleTimeoutMs = profile.streamIdleTimeoutMs
    using watchdog = idleWatchdog(upstream, streamIdleTimeoutMs, 'LLM_STREAM_IDLE_TIMEOUT')

    try {
      const containsImage = options.messages.some(message => contentHasImage(message.content))
      if (containsImage && !model.input.includes('image')) {
        throw new LlmError(`pi-ai model "${model.id}" does not support image input`, 'UNSUPPORTED_CONTENT')
      }
      const attachments = containsImage ? this.config.resolveAttachments?.() : undefined
      if (containsImage && attachments === undefined) {
        throw new LlmError('pi-ai image input requires the durable attachment service', 'UNSUPPORTED_CONTENT')
      }
      const onReplayDegrade = (reason: string): void => {
        this.config.onReplayDegrade?.({ provider: options.provider, model: options.model, reason })
      }
      const prefixDigests: PrefixDigests = new Map()
      const converted = attachments === undefined
        ? toPiContext(request, undefined, onReplayDegrade, updates.systemPromptUpdate === 'in-history', prefixDigests)
        : await toPiContext({ ...request, signal: watchdog.signal }, {
          attachments,
          resolveImageAccess: ref => this.config.resolveImageAccess?.(attachments, ref),
          maxRequestImageBytes: profile.maxRequestImageBytes,
          maxRequestImages: profile.maxRequestImages,
          imageOffloadCountQuantum: profile.imageOffloadCountQuantum,
          requestImagePolicy: resolveImageRequestBudget(model, profile),
        }, onReplayDegrade, updates.systemPromptUpdate === 'in-history', prefixDigests)
      const manualThinking = model.reasoning && reasoning !== undefined && reasoning !== 'off'
        && (model as Model<'anthropic-messages'>).compat?.forceAdaptiveThinking !== true
      const { context, digest } = bindThinkingToPrefix(converted, model.api, prefixDigests, manualThinking, (count) => {
        this.config.onThinkingOmitted?.({ provider: options.provider, model: options.model, count })
      })
      const events = snapshot.models.streamSimple(model, context, {
        ...profileOptions(profile, reasoning, apiKey),
        ...options.temperature === undefined ? {} : { temperature: options.temperature },
        ...options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens },
        ...options.sessionId === undefined ? {} : { sessionId: String(options.sessionId) },
        signal: watchdog.signal,
        headers: requestHeaders(profile.headers, options, model.baseUrl, this.config.prepareHeaders),
      })
      const iterator = toStreamChunks(events, model.contextWindow, options.signal, model.id, digest)[Symbol.asyncIterator]()
      let exhausted = false
      try {
        while (true) {
          const result = await watchdog.next(iterator)
          if (result.done) {
            exhausted = true
            return
          }
          yield result.value
        }
      } finally {
        if (!exhausted) {
          consumer.abort('pi-ai stream consumer stopped')
          // A demand the deadline abandoned can stay pending forever, and this
          // generator queues its return behind it; the abort above owns SDK termination.
          /* v8 ignore next -- the detached drain only rejects while unwinding an abandoned stream. */
          void iterator.return(undefined).catch(() => {})
        }
      }
    } catch (error: unknown) {
      if (timeoutOf(watchdog.signal, 'LLM_STREAM_IDLE_TIMEOUT') !== undefined) {
        throw new LlmError(`pi-ai stream idle timeout after ${streamIdleTimeoutMs}ms`, 'TIMEOUT', { cause: error })
      }
      if (options.signal?.aborted) {
        throw new LlmError('pi-ai request aborted by caller', 'ABORTED', { cause: error })
      }
      throw error
    } finally {
      consumer.abort('pi-ai stream consumer stopped')
    }
  }
}
