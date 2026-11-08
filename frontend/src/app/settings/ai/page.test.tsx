import { render, screen } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";

import AiProvidersPage from "./page";

vi.mock("@/lib/auth-client", () => ({
  useSession: () => ({ data: { user: { id: "user-1", email: "owner@example.com" } }, isPending: false }),
}));

const payload = {
  mode: "free",
  mode_label: "Free mode",
  mode_description: "Agnes AI for clip reasoning, Faster-Whisper for local transcription.",
  modes: [
    { id: "free", label: "Free mode", description: "Agnes AI + Faster-Whisper", selected: true },
    { id: "premium", label: "Premium mode", description: "Your paid provider", selected: false },
  ],
  reasoning: {
    provider: "agnes",
    model: "agnes:agnes-2.5-flash",
    label: "Agnes AI",
    tier: "free",
    configured: true,
  },
  transcription: {
    provider: "faster_whisper",
    label: "Faster-Whisper (local)",
    tier: "free",
    configured: true,
  },
  requires_paid_provider: false,
  paid_providers_authorized: false,
  free_fallback: { provider: "ollama", model: "ollama:gpt-oss:20b", configured: true },
  providers: [
    {
      id: "agnes",
      label: "Agnes AI",
      tier: "free",
      kind: "reasoning",
      description: "Default free reasoning provider.",
      configured: true,
      selected: true,
      paid: false,
      authorized: true,
    },
    {
      id: "openai",
      label: "OpenAI",
      tier: "paid",
      kind: "reasoning",
      description: "Premium reasoning provider.",
      configured: false,
      selected: false,
      paid: true,
      authorized: false,
    },
    {
      id: "faster_whisper",
      label: "Faster-Whisper (local)",
      tier: "free",
      kind: "transcription",
      description: "Local transcription with word timestamps.",
      configured: true,
      selected: true,
      paid: false,
      authorized: true,
    },
  ],
  warnings: ["Paid provider credentials are present but the platform is in Free mode."],
  transcription_settings: {
    engine: "faster_whisper",
    model_size: "medium",
    device: "cpu",
    compute_type: "int8",
    cache_enabled: true,
  },
  usage: {
    scope: "process",
    note: "Counts reported by providers since this process started.",
    entries: [
      { provider: "agnes", model: "agnes-2.5-flash", tier: "free", requests: 2, total_tokens: 320, errors: 0 },
    ],
  },
};

beforeEach(() => {
  vi.unstubAllGlobals();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), { status: 200 })),
  );
});

it("shows the free mode, the selected providers and paid-provider status", async () => {
  render(<AiProvidersPage />);

  expect(await screen.findByRole("heading", { name: "Free mode" })).toBeInTheDocument();
  expect(screen.getByText("No paid providers")).toBeInTheDocument();
  expect(screen.getByText("agnes:agnes-2.5-flash")).toBeInTheDocument();
  // The summary and the provider list both name the engine.
  expect(screen.getAllByText("Faster-Whisper (local)").length).toBeGreaterThan(0);
  expect(screen.getByText("Free fallback ready: ollama:gpt-oss:20b")).toBeInTheDocument();
  expect(screen.getByText(/Paid provider credentials are present/)).toBeInTheDocument();
  expect(screen.getByText("Blocked in this mode")).toBeInTheDocument();
  expect(screen.getByRole("table")).toHaveTextContent("320");
  expect(fetch).toHaveBeenCalledWith("/api/ai-providers", { cache: "no-store" });
});

it("reports paid usage when the deployment runs a premium provider", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ ...payload, mode: "premium", mode_label: "Premium mode", requires_paid_provider: true }),
        { status: 200 },
      ),
    ),
  );

  render(<AiProvidersPage />);

  expect(await screen.findByText("Paid provider in use")).toBeInTheDocument();
});

it("surfaces a load failure with a retry action", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("nope", { status: 500 })));

  render(<AiProvidersPage />);

  expect(await screen.findByText("Could not load AI provider settings.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
});
