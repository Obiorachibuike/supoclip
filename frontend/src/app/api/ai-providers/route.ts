import { NextResponse } from "next/server";

import { createTextProxyResponse, fetchBackend } from "@/server/backend-api";
import { getServerSession } from "@/server/session";

/**
 * Proxy the backend's read-only AI provider status.
 *
 * Keys never leave the backend: the response only reports which providers are
 * configured, which one is selected, and whether a task would spend money.
 */
export async function GET() {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const upstream = await fetchBackend("/ai/providers", {
    method: "GET",
    userId: session.user.id,
    cache: "no-store",
  });

  return createTextProxyResponse(upstream);
}
