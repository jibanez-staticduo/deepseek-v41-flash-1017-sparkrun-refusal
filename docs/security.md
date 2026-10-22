# Security

## Inference network

The inference API is configured to listen on the container's network interfaces. This repository's refusal feature does not add authentication to ordinary inference requests. Put the API behind an independently configured network boundary and authentication layer before exposing it beyond a trusted network.

## Refusal administration

When the optional admin route is enabled, `GET` and `POST /admin/refusal_lambda` are restricted by the application to loopback client addresses and require a bearer token. The token is read from a mounted file named by `STATICDUO_REFUSAL_ADMIN_TOKEN_FILE`.

Provision a random token outside the image, mount its file read-only with owner-only permissions (for example, mode `0600`), and keep it out of source control, command history, container build arguments, logs, and support bundles. The application does not create or rotate this credential for you.

For remote administration, use an SSH tunnel that terminates on the inference host and connects to its loopback address. A reverse proxy must not expose the admin route. Do not use `X-Forwarded-For` as an authorization signal: the route checks the direct peer address. Ensure the proxy, firewall, and container network policy prevent external direct access to the admin path.

The loopback and token checks protect this admin operation only. They do not protect port 8000 inference traffic or replace host/network isolation.

## Request and cache isolation

The effective lambda is frozen when a request is admitted. A global change affects future admissions; it does not change in-flight requests. The internal cache identity includes the exact float value, direction artifact SHA, and caller-provided tenant salt. Clients may override the default per request with an explicit refusal cache salt. Do not treat this field as authorization or as a tenant identity unless the surrounding service assigns and validates tenant salts.

Fallback to another model changes the backend and does not carry the DeepSeek projection semantics with it. Document the policy of each fallback model separately; do not describe a fallback as preserving this refusal control.

## Supply chain and data

Keep model weights and runtime secrets outside the image. Verify the upstream source revision, patch file hashes, direction artifact checksum, and immutable image digest. Do not publish private logs, local hostnames, network addresses, filesystem paths, or request content in build receipts or issue reports.
