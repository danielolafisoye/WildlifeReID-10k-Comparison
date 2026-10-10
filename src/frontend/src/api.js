// Thin client for the Flask REST API. In dev the Vite proxy forwards /api to localhost:5000.
// For a deployed build, set VITE_API_URL (e.g. https://reid.example.com).
const BASE = import.meta.env.VITE_API_URL ?? "";

async function parse(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    // non-JSON error page
  }
  if (!res.ok) {
    throw new Error(body?.error ?? `Request failed (${res.status})`);
  }
  return body;
}

export async function getHealth() {
  return parse(await fetch(`${BASE}/api/health`));
}

export async function getConfig() {
  return parse(await fetch(`${BASE}/api/config`));
}

export async function predict(file, threshold) {
  const form = new FormData();
  form.append("image", file);
  if (threshold !== undefined && threshold !== null) {
    form.append("threshold", String(threshold));
  }
  return parse(await fetch(`${BASE}/api/predict`, { method: "POST", body: form }));
}
