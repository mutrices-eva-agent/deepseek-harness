import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { LlmError } from '@deepseek-ai/dsh-llm'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { PiAiAdapter } from '@deepseek-ai/dsh-llm-pi-ai'
import type { Api, Model } from '@earendil-works/pi-ai'
import { getBuiltinModels } from '@earendil-works/pi-ai/providers/all'
import { resolveProfiles } from '../src/config.ts'
import { syncProviderModels } from '../src/sync.ts'
import { assemble } from './assemble.ts'
import { memoryAuth } from './auth-double.ts'
import { closeMockServers, mockServer, textEvents } from './mock-server.ts'

beforeEach(() => {
  vi.stubEnv('PI_SYNC_KEY', 'test-key')
})

afterEach(async () => {
  vi.useRealTimers()
  vi.unstubAllEnvs()
  await closeMockServers()
})

async function harness(
  server: { url: string },
  profile: LlmPiAi.PiAiProviderProfile = {},
): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(LlmPiAi, {
    providers: { deepseek: { apiKeyEnv: 'PI_SYNC_KEY', baseURL: server.url, ...profile } },
  })
  return ctx
}

function listing(...models: Record<string, unknown>[]): { body: string } {
  return { body: JSON.stringify({ data: models }) }
}

describe('automatic provider model synchronization', () => {
  it('advertises a newly listed model and uses it for a request', async () => {
    const server = await mockServer([
      listing({ id: 'deepseek-v5-next', display_name: 'DeepSeek V5 Next', context_length: 123_000, max_output_tokens: 8_000 }),
      { events: textEvents },
    ])
    const ctx = await harness(server, {
      api: 'openai-completions',
      compat: { supportsFinishReason: false },
    })
    const changed = vi.fn()
    ctx.on('llm/models-updated', changed)

    const models = await ctx.llm.listModels('deepseek')
    expect(models.find(model => model.id === 'deepseek-v5-next')).toEqual({
      provider: 'deepseek', id: 'deepseek-v5-next', name: 'DeepSeek V5 Next', inputModalities: ['text'],
    })
    expect(models.find(model => model.id === 'deepseek-flash')).toBeDefined()
    expect(changed).toHaveBeenCalledOnce()

    const prepared = await ctx.llm.prepareCall({ provider: 'deepseek', model: 'deepseek-v5-next' })
    expect(prepared.context).toEqual({ contextWindow: 123_000 })
    expect(prepared.config.maxTokens).toBeUndefined()
    const result = await assemble(ctx, { model: 'deepseek-v5-next', messages: [], purpose: 'session-title' })
    expect(result.message.content).toEqual([{ type: 'text', text: 'hello' }])
    expect(server.paths).toEqual(['/models', '/chat/completions'])
    expect(server.headers[0]?.authorization).toBe('Bearer test-key')
  })

  it('resolves an unknown model through the listing without a preceding catalog read', async () => {
    const server = await mockServer([
      listing({ id: 'deepseek-v5-next', context_length: 123_000 }),
    ])
    const ctx = await harness(server)

    await expect(ctx.llm.resolveModelInfo('deepseek', 'deepseek-v5-next')).resolves.toMatchObject({
      id: 'deepseek-v5-next', name: 'deepseek-v5-next', context: { contextWindow: 123_000 },
    })
    expect(server.paths).toEqual(['/models'])
  })

  it('keeps the last good catalog after a failed refresh and publishes the next change', async () => {
    const server = await mockServer([
      listing({ id: 'deepseek-v5-next' }),
      { status: 500, body: '{"error":"unavailable"}' },
      listing({ id: 'deepseek-v5-next' }, { id: 'deepseek-v5-later' }),
    ])
    const ctx = await harness(server)
    const changed = vi.fn()
    ctx.on('llm/models-updated', changed)

    expect((await ctx.llm.listModels('deepseek')).some(model => model.id === 'deepseek-v5-next')).toBe(true)
    expect((await ctx.llm.listModels('deepseek')).some(model => model.id === 'deepseek-v5-next')).toBe(true)
    const latest = await ctx.llm.listModels('deepseek')

    expect(latest.some(model => model.id === 'deepseek-v5-later')).toBe(true)
    expect(changed).toHaveBeenCalledTimes(2)
    expect(server.paths).toEqual(['/models', '/models', '/models'])
  })

  it('keeps explicit models authoritative without contacting the endpoint', async () => {
    const server = await mockServer([])
    const ctx = await harness(server, { models: [{ id: 'only-mine' }] })

    expect((await ctx.llm.listModels('deepseek')).map(model => model.id)).toEqual(['only-mine'])
    await expect(ctx.llm.resolveModelInfo('deepseek', 'only-mine')).resolves.toMatchObject({ id: 'only-mine' })
    expect(server.paths).toEqual([])
  })

  it('applies installed-model metadata and explicit overrides above listing fields', async () => {
    const installed = getBuiltinModels('deepseek').find(model => model.id === 'deepseek-flash')
    if (installed === undefined) throw new Error('missing installed deepseek-flash fixture')
    const server = await mockServer([
      listing({ id: installed.id, display_name: 'Provider name', context_length: 99_000, max_output_tokens: 9_000 }),
    ])
    const ctx = await harness(server, {
      modelOverrides: { [installed.id]: { name: 'Deployment name', maxTokens: 4_096 } },
    })

    const listed = (await ctx.llm.listModels('deepseek')).find(model => model.id === installed.id)
    const resolved = await ctx.llm.resolveModelInfo('deepseek', installed.id)
    expect(listed).toMatchObject({ name: 'Deployment name', inputModalities: installed.input })
    expect(resolved).toMatchObject({
      name: 'Deployment name', context: { contextWindow: 99_000 }, defaultMaxTokens: 4_096,
    })
    expect(resolved.reasoning?.efforts.map(effort => effort.id)).toEqual(['off', 'low', 'high', 'max'])
  })
})

describe('provider listing capabilities', () => {
  it('uses the catalog protocol for a newly listed OpenAI model', async () => {
    const server = await mockServer([listing({ id: 'gpt-sync-unlisted' })])
    const profile = resolveProfiles({ openai: { baseURL: server.url } }).get('openai')!
    const installed = profile.piProvider!.getModels()
    const responses = installed.find(model => model.api === 'openai-responses')
    if (responses === undefined) throw new Error('missing mixed-protocol OpenAI fixture')

    const synced = await syncProviderModels('openai', profile, undefined)

    expect(synced?.find(model => model.id === 'gpt-sync-unlisted')).toMatchObject({ api: 'openai-responses' })
    expect(synced?.find(model => model.id === responses.id)).toMatchObject({ api: 'openai-responses' })
    expect(server.paths).toEqual(['/models'])
  })

  it('retains routes without an unambiguous supported listing', async () => {
    const azure = resolveProfiles({ azure: {} }).get('azure')!
    const mistral = resolveProfiles({ mistral: {} }).get('mistral')!
    const missing = resolveProfiles({ 'retired-route': {} }, 'deferred').get('retired-route')!

    await expect(syncProviderModels('azure', azure, 'test-key')).resolves.toBeUndefined()
    await expect(syncProviderModels('mistral', mistral, 'test-key')).resolves.toBeUndefined()
    await expect(syncProviderModels('retired-route', missing, 'test-key')).resolves.toBeUndefined()
  })
})

describe('provider synchronization concurrency and deadlines', () => {
  function directAdapter(
    syncModels: NonNullable<ConstructorParameters<typeof PiAiAdapter>[0]['syncModels']>,
    onModelsUpdated?: () => void,
    onModelSyncError?: (error: unknown) => void,
  ): PiAiAdapter {
    const profiles = resolveProfiles({ deepseek: { modelsTimeoutMs: 20 } })
    return new PiAiAdapter({
      profiles: () => profiles,
      resolveApiKey: () => Promise.resolve('test-key'),
      syncModels,
      ...onModelsUpdated === undefined ? {} : { onModelsUpdated },
      ...onModelSyncError === undefined ? {} : { onModelSyncError: ({ error }) => { onModelSyncError(error) } },
      auth: memoryAuth(),
    })
  }

  it('shares one in-flight listing and publishes one changed catalog', async () => {
    const profile = resolveProfiles({ deepseek: {} }).get('deepseek')!
    const baseline = profile.piProvider!.getModels()
    const added: Model<Api> = { ...baseline[0]!, id: 'deepseek-v5-next', name: 'DeepSeek V5 Next' }
    const pending = Promise.withResolvers<readonly Model<Api>[]>()
    const started = Promise.withResolvers<undefined>()
    const load = vi.fn(() => {
      started.resolve(undefined)
      return pending.promise
    })
    const changed = vi.fn()
    const adapter = directAdapter(load, changed)

    const first = adapter.listModels('deepseek')
    const second = adapter.listModels('deepseek')
    await started.promise
    expect(load).toHaveBeenCalledOnce()
    pending.resolve([...baseline, added])

    expect((await first).some(model => model.id === added.id)).toBe(true)
    expect(await second).toEqual(await first)
    expect(changed).toHaveBeenCalledOnce()
  })

  it('lets one caller abandon a shared listing wait', async () => {
    const pending = Promise.withResolvers<readonly Model<Api>[]>()
    const started = Promise.withResolvers<undefined>()
    const adapter = directAdapter(() => {
      started.resolve(undefined)
      return pending.promise
    })
    const listing = adapter.listModels('deepseek')
    const controller = new AbortController()
    const resolving = adapter.resolveModel('deepseek', 'deepseek-v5-next', controller.signal)
    await started.promise
    controller.abort('caller left')
    await expect(resolving).rejects.toBe('caller left')

    const profile = resolveProfiles({ deepseek: {} }).get('deepseek')!
    pending.resolve(profile.piProvider!.getModels())
    await expect(listing).resolves.not.toHaveLength(0)

    const aborted = AbortSignal.abort('already gone')
    await expect(adapter.resolveModel('deepseek', 'deepseek-v5-next', aborted)).rejects.toBe('already gone')
  })

  it('ignores an unavailable listing and a result from a superseded configuration', async () => {
    const unavailable = directAdapter(() => Promise.resolve(undefined))
    await expect(unavailable.listModels('deepseek')).resolves.toHaveLength(getBuiltinModels('deepseek').length)

    let profiles = resolveProfiles({ deepseek: {} })
    const baseline = profiles.get('deepseek')!.piProvider!.getModels()
    const added: Model<Api> = { ...baseline[0]!, id: 'stale-model' }
    const pending = Promise.withResolvers<readonly Model<Api>[]>()
    const started = Promise.withResolvers<undefined>()
    const changed = vi.fn()
    const adapter = new PiAiAdapter({
      profiles: () => profiles,
      resolveApiKey: () => Promise.resolve('test-key'),
      syncModels: () => {
        started.resolve(undefined)
        return pending.promise
      },
      onModelsUpdated: changed,
      auth: memoryAuth(),
    })
    const listing = adapter.listModels('deepseek')
    await started.promise
    profiles = resolveProfiles({ deepseek: { baseURL: 'https://replacement.test' } })
    adapter.providerInfo('deepseek')
    pending.resolve([...baseline, added])

    expect((await listing).some(model => model.id === added.id)).toBe(false)
    expect(changed).not.toHaveBeenCalled()
  })

  it('bounds a stalled listing and retains the installed catalog', async () => {
    const failure = vi.fn()
    const adapter = directAdapter(
      (_provider, _profile, _apiKey, signal) => new Promise((_resolve, reject) => {
        signal.addEventListener('abort', () => {
          reject(new LlmError('listing deadline', 'DISCOVERY_FAILED'))
        }, { once: true })
      }),
      undefined,
      failure,
    )

    await expect(adapter.listModels('deepseek')).resolves.toHaveLength(getBuiltinModels('deepseek').length)
    expect(failure).toHaveBeenCalledOnce()
  })
})
