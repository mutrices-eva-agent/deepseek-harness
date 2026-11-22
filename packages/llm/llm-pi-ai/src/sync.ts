/** Provider-listed model catalogs layered over the installed pi-ai catalog. */

import type { Api, Model } from '@earendil-works/pi-ai'
import { resolveRouteModels } from './catalog.ts'
import type { PiAiModelProfile } from './catalog.ts'
import {
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_INPUT,
  DEFAULT_MAX_TOKENS,
} from './config.ts'
import type { ResolvedPiAiProviderProfile } from './config.ts'
import { discoverModels, supportsModelDiscovery } from './discovery.ts'

/** Select the listing protocol without changing existing models' individual protocols. */
function listingApi(profile: ResolvedPiAiProviderProfile): string | undefined {
  if (profile.api !== undefined) return profile.api
  const apis = new Set(profile.piProvider?.getModels().map(model => model.api))
  if (apis.size === 1) return [...apis][0]
  return undefined
}

/**
 * Load a route's latest provider listing and merge it with its configured catalog.
 * Installed models supply protocol and capability metadata; the listing updates
 * capacities and appends new models. Existing models not present in a listing
 * remain usable. Explicit model overrides win over listing fields.
 * @param provider - provider route being refreshed.
 * @param profile - resolved route configuration without a replacing models list.
 * @param apiKey - resolved credential, or undefined for ambient/unauthenticated discovery.
 * @param signal - optional cancellation and deadline for the listing request.
 * @returns the full catalog to serve, or undefined when this route has no supported listing endpoint.
 */
export async function syncProviderModels(
  provider: string,
  profile: ResolvedPiAiProviderProfile,
  apiKey: string | undefined,
  signal?: AbortSignal,
): Promise<readonly Model<Api>[] | undefined> {
  const current = profile.piProvider
  const baseURL = profile.baseURL ?? current?.baseUrl
  const api = listingApi(profile)
  if (current === undefined || baseURL === undefined || api === undefined || !supportsModelDiscovery(api)) return undefined

  const discovered = await discoverModels({
    baseURL,
    api,
    ...apiKey === undefined ? {} : { apiKey },
    ...signal === undefined ? {} : { signal },
  }, () => ({ headers: profile.headers, resolveApiKey: () => Promise.resolve(apiKey) }))
  const overrides = profile.modelOverrides ?? {}
  const entries: PiAiModelProfile[] = discovered.map(model => ({
    id: model.id,
    ...model.name === model.id ? {} : { name: model.name },
    ...model.contextWindow === undefined ? {} : { contextWindow: model.contextWindow },
    ...model.maxTokens === undefined ? {} : { maxTokens: model.maxTokens },
    ...overrides[model.id],
  }))
  const resolved = resolveRouteModels({
    provider,
    ...profile.api === undefined ? {} : { api: profile.api },
    baseURL,
    models: entries,
    ...profile.compat === undefined ? {} : { compat: profile.compat },
    defaultContextWindow: profile.defaultContextWindow ?? DEFAULT_CONTEXT_WINDOW,
    defaultMaxTokens: profile.defaultMaxTokens ?? DEFAULT_MAX_TOKENS,
    defaultInput: profile.defaultInput ?? [...DEFAULT_INPUT],
  })
  const merged = new Map(current.getModels().map(model => [model.id, model]))
  for (const model of resolved.models) merged.set(model.id, model)
  return [...merged.values()]
}
