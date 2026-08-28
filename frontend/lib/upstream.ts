export const API_BASE = (
  process.env.AGENTIQ_API_URL ?? "http://23.21.42.197:8000"
).replace(/\/+$/, "");
