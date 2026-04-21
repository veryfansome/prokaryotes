# Compaction Indicator Dismissal

## Background

After a response that triggers compaction, `stream_and_finalize` emits a `{"compaction_pending": true}` NDJSON event. The UI shows a "Compressing context…" indicator and clears it when the next stream begins with a `partition_uuid` that differs from the one that triggered compaction. This is the correct semantic signal, but it requires the user to send another message. If the user reads the response and walks away, the indicator stays on screen indefinitely.

Two options are documented here.

---

## Option A — Client-side auto-timeout

### What it does

After `compaction_pending` is received, start a `setTimeout` that calls `clearCompactionIndicator()` after a fixed delay. If the user sends another message before the timer fires, the existing `partition_uuid` comparison clears the indicator at the correct semantic moment and the timeout becomes a no-op (`clearTimeout` is called inside `clearCompactionIndicator`).

### Changes required

`scripts/static/ui.js` only (~10 lines):

```js
const COMPACTION_INDICATOR_TIMEOUT_MS = 30_000;
let compactionIndicatorTimer = null;

function showCompactionIndicator(partitionUuid) {
    const el = doc.getElementById('compaction-indicator');
    if (el) { el.hidden = false; }
    compactionIndicatorVisible = true;
    pendingCompactionPartitionUuid = partitionUuid;
    clearTimeout(compactionIndicatorTimer);
    compactionIndicatorTimer = setTimeout(clearCompactionIndicator, COMPACTION_INDICATOR_TIMEOUT_MS);
}

function clearCompactionIndicator() {
    clearTimeout(compactionIndicatorTimer);
    const el = doc.getElementById('compaction-indicator');
    if (el) { el.hidden = true; }
    compactionIndicatorVisible = false;
    pendingCompactionPartitionUuid = null;
}
```

No server changes. No new tests beyond updating the existing `showCompactionIndicator` JS test to assert the timer fires.

### Trade-offs

- Pros: trivial to implement; no network traffic; no new endpoint.
- Cons: the timeout is a guess. If compaction finishes in 5 seconds the indicator lingers for 25 more. If compaction takes longer than the timeout (large context, slow model) the indicator clears before the swap is done. Neither case is harmful — the indicator is cosmetic — but the UX is imprecise.

A 30-second default covers typical summarization latency with margin and can be tuned down to 15 seconds as empirical data accumulates.

---

## Option B — Server-side polling endpoint

### What it does

After `compaction_pending` is received, the client starts a `setInterval` that fires every 15 seconds. Each tick sends a `GET /compaction-status?conversation_uuid=…&pending_partition_uuid=…`. The server checks two Redis keys and returns `{"done": true|false}`. The client stops polling and clears the indicator when `done` is `true`, when the user sends another message (which already clears the indicator), or on a network error.

### Changes required

**`prokaryotes/web_v1.py`** — new method and route registration in `WebBase` (~25 lines):

```python
async def get_compaction_status(
        self,
        request: Request,
        conversation_uuid: str,
        pending_partition_uuid: str,
):
    await load_session(request)
    if not request.session:
        raise HTTPException(status_code=400, detail="Session expired")
    lock_exists = await self.redis_client.exists(
        f"compaction_lock:{conversation_uuid}"
    )
    if lock_exists:
        return {"done": False}
    cached = await self.redis_client.get(
        f"context_partition:{conversation_uuid}"
    )
    if cached:
        partition = ContextPartition.model_validate_json(cached)
        return {"done": partition.partition_uuid != pending_partition_uuid}
    return {"done": True}  # Redis evicted — compaction definitely finished
```

Register in `WebBase.init()`:
```python
self.app.add_api_route(
    "/compaction-status", self.get_compaction_status, methods=["GET"]
)
```

**`scripts/static/ui.js`** — polling loop (~35 lines):

```js
let compactionPollInterval = null;

function startCompactionPolling(conversationUuid, pendingPartitionUuid) {
    stopCompactionPolling();
    compactionPollInterval = setInterval(async () => {
        try {
            const params = new URLSearchParams({
                conversation_uuid: conversationUuid,
                pending_partition_uuid: pendingPartitionUuid,
            });
            const res = await fetchImpl(`/compaction-status?${params}`);
            if (!res.ok) { stopCompactionPolling(); return; }
            const data = await res.json();
            if (data.done) {
                clearCompactionIndicator();
                stopCompactionPolling();
            }
        } catch {
            stopCompactionPolling();
        }
    }, 15_000);
}

function stopCompactionPolling() {
    clearInterval(compactionPollInterval);
    compactionPollInterval = null;
}
```

Call `startCompactionPolling(conversationUuid, receivedPartitionUuid)` from the `compaction_pending` branch of `processLine`. Call `stopCompactionPolling()` wherever `clearCompactionIndicator()` is already called.

**Tests** — 2–3 new Python tests for the endpoint (lock present → not done; lock absent + new UUID → done; lock absent + same UUID → done). 1–2 new JS tests for the polling loop (mock a sequence of `done: false` followed by `done: true` and assert the indicator clears on the second tick).

### Trade-offs

- Pros: accurate — the indicator clears as soon as compaction finishes regardless of whether the user interacts; handles both fast and slow compactions correctly.
- Cons: adds a network round-trip every 15 seconds per active indicator; requires a new endpoint and corresponding tests; requires careful poll cleanup (page close, user sends message mid-poll, etc.).

The accuracy gain is most meaningful when compaction is expected to run long (large contexts, slow or high-latency models) and users are expected to sit and watch the indicator. For typical conversation lengths where summarization finishes in under 15 seconds, polling and the auto-timeout produce nearly identical UX.

---

## Recommendation

Start with **Option A**. The indicator is cosmetic — it tells the user that compression was scheduled, not that it is actively in progress — so precision is not load-bearing. Revisit **Option B** if empirical latency data shows compactions routinely exceeding the timeout or if user confusion around the indicator is reported.
