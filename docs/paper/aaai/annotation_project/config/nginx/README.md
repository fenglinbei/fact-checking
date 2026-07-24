# Label Studio nginx edge configuration

`fc-fenglin.conf` is the reviewed source for the public
`fc.fenglin.pro` virtual host. It keeps all dynamic Label Studio routes
uncached and only caches the public `/static/` and `/react-app/` asset
prefixes.

## Runtime paths

- Active config: `/etc/nginx/conf.d/fc-fenglin.conf` on SSH host `dig`
- Static cache: `/var/cache/nginx/label-studio-static`
- Pre-change backup:
  `/etc/nginx/backups/fc-reachability-20260724_220425/fc-fenglin.conf.before`

The cache key includes the full request URI, including the Label Studio
frontend version query. Cached assets are served for seven days and may
be served stale while the reverse tunnel is temporarily unavailable.
Login, API, task, annotation, and `/evidence-map/` responses explicitly
remain outside this cache.

## Change procedure

1. Back up the active config.
2. Install the candidate file without reloading nginx.
3. Run `sudo nginx -t`.
4. Gracefully reload nginx only after the syntax check succeeds.
5. Verify a static request transitions from `X-Cache-Status: MISS` to
   `HIT`, has no `Set-Cookie`, and preserves the origin asset SHA-256.
6. Verify a browser-equivalent login POST still returns the normal
   application response and has no `X-Cache-Status`.

After a Label Studio upgrade, clear and prewarm only the dedicated
`label-studio-static` cache after verifying that the new frontend asset
hashes match the upgraded local service.

## Rollback

Restore the pre-change config above, run `sudo nginx -t`, and gracefully
reload nginx. The cache directory is isolated and does not contain
Label Studio data or annotations.
