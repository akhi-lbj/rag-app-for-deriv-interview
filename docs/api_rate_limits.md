# API Rate Limits and Throttling Policy

## 1. Rate Limiting Tiers
All incoming API requests to the platform are subject to automated rate limiting based on the client's API tier:
- **Standard Tier**:
  - 60 requests per minute (RPM) per IP address / API key.
  - 1,000 requests per hour (RPH).
  - Burst allowance: up to 10 requests above RPM for spikes under 5 seconds.
- **Enterprise Tier**:
  - 600 requests per minute (RPM).
  - 50,000 requests per day.
  - Custom endpoint allotments available upon architectural review.

## 2. HTTP Headers Returned
Every API response includes real-time telemetry headers reflecting the client's quota status:
- `X-RateLimit-Limit`: The maximum number of requests allowed in the current time window.
- `X-RateLimit-Remaining`: The number of requests remaining in the current window.
- `X-RateLimit-Reset`: Unix epoch timestamp indicating when the current window resets.

## 3. Rate Limit Exceeded Behavior (HTTP 429)
- When a client breaches their allocated threshold, the API responds with **HTTP 429 Too Many Requests**.
- The response body contains an error payload: `{"error": "rate_limit_exceeded", "retry_after": <seconds>}`.
- Clients must respect the `Retry-After` response header and implement exponential backoff with randomized jitter to prevent thundering herd conditions.
