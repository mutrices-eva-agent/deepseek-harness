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

/** Authentication and endpoint facts resolved for one provider-listing operation. */
export interface ProviderListingAuth {
  /** Explicit or provider-native API key, when one is available. */
  apiKey?: string
  /** Provider-native authorization headers, excluding null-valued removals. */
  headers?: Readonly<Record<string, string>>
  /** Provider-native endpoint override, subordinate to the configured baseURL. */
  baseURL?: string
}

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
 * @param auth - explicit or provider-native authentication and endpoint facts.
 * @param signal - optional cancellation and deadline for the listing request.
 * @returns the full catalog to serve, or undefined when this route has no supported listing endpoint.
 */
export async function syncProviderModels(
  provider: string,
  profile: ResolvedPiAiProviderProfile,
  auth: ProviderListingAuth,
  signal?: AbortSignal,
): Promise<readonly Model<Api>[] | undefined> {
  const current = profile.piProvider
  const baseURL = profile.baseURL ?? auth.baseURL ?? current?.baseUrl
  const api = listingApi(profile)
  if (current === undefined || baseURL === undefined || api === undefined || !supportsModelDiscovery(api)) return undefined

  const discovered = await discoverModels({
    baseURL,
    api,
    ...auth.apiKey === undefined ? {} : { apiKey: auth.apiKey },
    ...signal === undefined ? {} : { signal },
  }, () => ({
    headers: { ...auth.headers, ...profile.headers },
    resolveApiKey: () => Promise.resolve(auth.apiKey),
  }))
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
