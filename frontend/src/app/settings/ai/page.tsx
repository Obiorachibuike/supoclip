"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  ArrowUpRight,
  Bot,
  Check,
  Cpu,
  DollarSign,
  Loader2,
  Mic,
  ShieldCheck,
} from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/lib/auth-client";

export interface AiProviderEntry {
  id: string;
  label: string;
  tier: "free" | "local" | "paid";
  kind: "reasoning" | "transcription";
  description: string;
  configured: boolean;
  selected: boolean;
  paid: boolean;
  authorized: boolean;
}

export interface AiProvidersPayload {
  mode: string;
  mode_label: string;
  mode_description: string;
  modes: { id: string; label: string; description: string; selected: boolean }[];
  reasoning: { provider: string; model: string; label: string; tier: string; configured: boolean };
  transcription: { provider: string; label: string; tier: string; configured: boolean };
  requires_paid_provider: boolean;
  paid_providers_authorized: boolean;
  free_fallback: { provider: string | null; model: string | null; configured: boolean };
  providers: AiProviderEntry[];
  warnings: string[];
  transcription_settings?: {
    engine: string;
    model_size: string | null;
    device: string | null;
    compute_type: string | null;
    cache_enabled: boolean;
  };
  usage?: {
    scope: string;
    note: string;
    entries: {
      provider: string;
      model: string;
      tier: string;
      requests: number;
      total_tokens: number;
      errors: number;
    }[];
  };
}

const tierBadge: Record<AiProviderEntry["tier"], { label: string; variant: "default" | "secondary" | "outline" }> = {
  free: { label: "Free", variant: "secondary" },
  local: { label: "Local", variant: "outline" },
  paid: { label: "Paid", variant: "default" },
};

export default function AiProvidersPage() {
  const { data: session, isPending } = useSession();
  const [payload, setPayload] = useState<AiProvidersPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isFetching, setIsFetching] = useState(true);

  const load = useCallback(async () => {
    setError(null);
    try {
      const response = await fetch("/api/ai-providers", { cache: "no-store" });
      if (!response.ok) {
        throw new Error("Could not load AI provider settings.");
      }
      setPayload((await response.json()) as AiProvidersPayload);
    } catch (loadError) {
      setError(
        loadError instanceof Error ? loadError.message : "Could not load AI provider settings.",
      );
    } finally {
      setIsFetching(false);
    }
  }, []);

  useEffect(() => {
    if (isPending) return;
    if (!session?.user) {
      setIsFetching(false);
      return;
    }
    void load();
  }, [isPending, session?.user, load]);

  if (isPending || isFetching) {
    return (
      <main className="mx-auto w-full max-w-4xl px-4 py-8">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="mt-4 h-40 w-full" />
        <span className="sr-only">Loading AI provider settings</span>
      </main>
    );
  }

  if (!session?.user) {
    return (
      <main className="mx-auto w-full max-w-4xl px-4 py-8">
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Sign in to view AI provider settings.</AlertDescription>
        </Alert>
      </main>
    );
  }

  if (error || !payload) {
    return (
      <main className="mx-auto w-full max-w-4xl px-4 py-8">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error ?? "AI provider settings are unavailable."}</AlertDescription>
        </Alert>
        <Button className="mt-4" variant="outline" onClick={() => void load()}>
          Try again
        </Button>
      </main>
    );
  }

  const reasoningProviders = payload.providers.filter((provider) => provider.kind === "reasoning");
  const transcriptionProviders = payload.providers.filter(
    (provider) => provider.kind === "transcription",
  );

  return (
    <main className="mx-auto w-full max-w-4xl px-4 py-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">AI providers</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Which models process your videos, and whether they cost money.
          </p>
        </div>
        <Button variant="outline" size="sm" asChild>
          <Link href="/settings">
            Back to settings
            <ArrowUpRight className="ml-1 h-4 w-4" />
          </Link>
        </Button>
      </header>

      <section
        aria-labelledby="mode-heading"
        className="mt-6 rounded-lg border border-border bg-card p-5"
      >
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-4 w-4" aria-hidden />
          <h2 id="mode-heading" className="text-lg font-medium">
            {payload.mode_label}
          </h2>
          <Badge variant={payload.requires_paid_provider ? "default" : "secondary"}>
            {payload.requires_paid_provider ? "Paid provider in use" : "No paid providers"}
          </Badge>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">{payload.mode_description}</p>
        <dl className="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <dt className="text-xs uppercase tracking-wide text-muted-foreground">
              Clip analysis
            </dt>
            <dd className="mt-1 flex items-center gap-2 text-sm">
              <Bot className="h-4 w-4" aria-hidden />
              <span className="font-medium">{payload.reasoning.label}</span>
              <code className="text-xs text-muted-foreground">{payload.reasoning.model}</code>
              {!payload.reasoning.configured ? (
                <Badge variant="destructive">Not configured</Badge>
              ) : null}
            </dd>
          </div>
          <div>
            <dt className="text-xs uppercase tracking-wide text-muted-foreground">
              Transcription
            </dt>
            <dd className="mt-1 flex items-center gap-2 text-sm">
              <Mic className="h-4 w-4" aria-hidden />
              <span className="font-medium">{payload.transcription.label}</span>
              {payload.transcription_settings ? (
                <code className="text-xs text-muted-foreground">
                  {payload.transcription_settings.model_size ?? ""} /{" "}
                  {payload.transcription_settings.device ?? ""}
                </code>
              ) : null}
              {!payload.transcription.configured ? (
                <Badge variant="destructive">Not configured</Badge>
              ) : null}
            </dd>
          </div>
        </dl>
        {payload.free_fallback.configured ? (
          <p className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
            <Cpu className="h-3.5 w-3.5" aria-hidden />
            Free fallback ready: {payload.free_fallback.model}
          </p>
        ) : null}
      </section>

      {payload.warnings.length > 0 ? (
        <section aria-labelledby="warnings-heading" className="mt-4 space-y-2">
          <h2 id="warnings-heading" className="sr-only">
            Warnings
          </h2>
          {payload.warnings.map((warning) => (
            <Alert key={warning}>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{warning}</AlertDescription>
            </Alert>
          ))}
        </section>
      ) : null}

      <section aria-labelledby="modes-heading" className="mt-6">
        <h2 id="modes-heading" className="text-lg font-medium">
          Processing modes
        </h2>
        <ul className="mt-3 grid gap-3 sm:grid-cols-2">
          {payload.modes.map((mode) => (
            <li
              key={mode.id}
              className="rounded-lg border border-border p-4"
              aria-current={mode.selected ? "true" : undefined}
            >
              <div className="flex items-center gap-2">
                <span className="font-medium">{mode.label}</span>
                {mode.selected ? (
                  <Badge variant="secondary">
                    <Check className="mr-1 h-3 w-3" aria-hidden />
                    Active
                  </Badge>
                ) : null}
              </div>
              <p className="mt-1 text-xs text-muted-foreground">{mode.description}</p>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-xs text-muted-foreground">
          Modes are configured by the operator with{" "}
          <code>AI_PROVIDER_MODE</code> (environment or admin runtime settings).
        </p>
      </section>

      <section aria-labelledby="providers-heading" className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <h2 id="providers-heading" className="text-lg font-medium">
            Clip analysis providers
          </h2>
          <ProviderList providers={reasoningProviders} />
        </div>
        <div>
          <h2 className="text-lg font-medium">Transcription providers</h2>
          <ProviderList providers={transcriptionProviders} />
        </div>
      </section>

      <section aria-labelledby="usage-heading" className="mt-6 rounded-lg border border-border p-5">
        <div className="flex items-center gap-2">
          <DollarSign className="h-4 w-4" aria-hidden />
          <h2 id="usage-heading" className="text-lg font-medium">
            Usage this process reported
          </h2>
        </div>
        {payload.usage && payload.usage.entries.length > 0 ? (
          <>
            <table className="mt-3 w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                  <th scope="col" className="py-1">Provider</th>
                  <th scope="col" className="py-1">Tier</th>
                  <th scope="col" className="py-1">Requests</th>
                  <th scope="col" className="py-1">Tokens</th>
                  <th scope="col" className="py-1">Errors</th>
                </tr>
              </thead>
              <tbody>
                {payload.usage.entries.map((entry) => (
                  <tr key={`${entry.provider}-${entry.model}`} className="border-t border-border">
                    <td className="py-1">{entry.provider}</td>
                    <td className="py-1">{entry.tier}</td>
                    <td className="py-1">{entry.requests}</td>
                    <td className="py-1">{entry.total_tokens}</td>
                    <td className="py-1">{entry.errors}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-xs text-muted-foreground">{payload.usage.note}</p>
          </>
        ) : (
          <p className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            No provider usage recorded by this process yet.
          </p>
        )}
      </section>
    </main>
  );
}

function ProviderList({ providers }: { providers: AiProviderEntry[] }) {
  return (
    <ul className="mt-3 space-y-2">
      {providers.map((provider) => {
        const badge = tierBadge[provider.tier] ?? tierBadge.paid;
        return (
          <li key={provider.id} className="rounded-lg border border-border p-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium">{provider.label}</span>
              <Badge variant={badge.variant}>{badge.label}</Badge>
              {provider.selected ? <Badge variant="secondary">Selected</Badge> : null}
              {!provider.configured ? (
                <Badge variant="outline">No credentials</Badge>
              ) : null}
              {provider.paid && !provider.authorized ? (
                <Badge variant="outline">Blocked in this mode</Badge>
              ) : null}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{provider.description}</p>
          </li>
        );
      })}
    </ul>
  );
}
