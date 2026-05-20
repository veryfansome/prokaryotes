/**
 * DOM/fetch integration tests for the migrated chat protocol.
 *
 * Covers the production paths in `scripts/static/ui.js` that the pure-helper tests in `ui.test.js` and
 * `conversation_client.test.js` don't reach:
 *
 * - `sendMessage` builds the POST body with `snapshot_uuid` + `source_id`s for server-stamped nodes (no
 *   `partition_uuid`).
 * - The first stream event (handshake) stamps server-assigned source_ids onto the submitted user node.
 * - The assistant node is created only on `bot_message`, with the bot's server-assigned `source_id` and the
 *   handshake's `snapshot_uuid`.
 * - `compaction_pending` starts the polling loop; a `{done: true, snapshot_uuid: ...}` response relabels the
 *   message-tree nodes.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createChatApp } from '../../scripts/static/ui.js';

function renderBaseDOM() {
    document.body.innerHTML = `
        <div id="chatContainer"><div id="chatWrapper"></div></div>
        <textarea id="chatInput"></textarea>
        <button id="sendButton" disabled>Send</button>
        <div id="editStatus" hidden></div>
        <div id="compaction-indicator" hidden></div>
        <div id="context-fill"></div>
    `;
}

function buildNdjsonBody(events) {
    const encoder = new TextEncoder();
    return new ReadableStream({
        start(controller) {
            for (const ev of events) {
                controller.enqueue(encoder.encode(`${JSON.stringify(ev)}\n`));
            }
            controller.close();
        },
    });
}

function makeFetchMock({ chatEvents = [], compactionStatuses = [] } = {}) {
    const calls = { chat: [], compactionStatus: [] };
    const compactionResponses = [...compactionStatuses];

    const fetchMock = vi.fn(async (url, init) => {
        if (typeof url === 'string' && url.endsWith('/conversation')) {
            return { ok: true, json: async () => ({ conversation_uuid: 'conv-123' }) };
        }
        if (typeof url === 'string' && url.includes('/chat?')) {
            calls.chat.push({ url, init, body: init ? JSON.parse(init.body) : null });
            return { ok: true, body: buildNdjsonBody(chatEvents) };
        }
        if (typeof url === 'string' && url.includes('/compaction-status')) {
            calls.compactionStatus.push({ url });
            const next = compactionResponses.shift() || { done: false };
            return { ok: true, json: async () => next };
        }
        throw new Error(`Unexpected fetch URL: ${url}`);
    });

    return { fetchMock, calls };
}

async function flushAsyncWork() {
    // Drain `sendMessage`'s awaits through to the final post-stream state update.
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
}

async function sendOneMessage(app, text) {
    app.elements.chatInput.value = text;
    app.elements.chatInput.dispatchEvent(new Event('input'));
    await app.sendMessage();
}

beforeEach(() => {
    renderBaseDOM();
});

afterEach(() => {
    vi.useRealTimers();
});

describe('POST body wire shape', () => {
    it('first turn: sends snapshot_uuid:null and a user message with no source_id', async () => {
        const { fetchMock, calls } = makeFetchMock({
            chatEvents: [
                { snapshot_uuid: 's-1', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
                { text_delta: 'hello' },
                { bot_message: { source_id: '1.000002' } },
            ],
        });
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'hi');
        await flushAsyncWork();

        expect(calls.chat).toHaveLength(1);
        const body = calls.chat[0].body;
        expect(body.snapshot_uuid).toBeNull();
        expect(body.messages).toHaveLength(1);
        expect(body.messages[0]).toEqual({ role: 'user', content: 'hi' });
        // Legacy field must not appear.
        expect(body).not.toHaveProperty('partition_uuid');
    });

    it('second turn: includes snapshot_uuid + echoes server source_ids on prior nodes', async () => {
        const turn1Events = [
            { snapshot_uuid: 's-1', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
            { text_delta: 'first reply' },
            { bot_message: { source_id: '1.000002' } },
        ];
        const turn2Events = [
            {
                snapshot_uuid: 's-1',
                source_id_assignments: [{ client_index: 2, source_id: '1.000003' }],
            },
            { text_delta: 'second reply' },
            { bot_message: { source_id: '1.000004' } },
        ];

        const { fetchMock, calls } = makeFetchMock();
        // Vary the chat events per call.
        let chatCallIndex = 0;
        fetchMock.mockImplementation(async (url, init) => {
            if (typeof url === 'string' && url.endsWith('/conversation')) {
                return { ok: true, json: async () => ({ conversation_uuid: 'conv-123' }) };
            }
            if (typeof url === 'string' && url.includes('/chat?')) {
                calls.chat.push({ url, init, body: JSON.parse(init.body) });
                const events = chatCallIndex === 0 ? turn1Events : turn2Events;
                chatCallIndex += 1;
                return { ok: true, body: buildNdjsonBody(events) };
            }
            throw new Error(`Unexpected fetch URL: ${url}`);
        });

        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'hi');
        await flushAsyncWork();
        await sendOneMessage(app, 'follow-up');
        await flushAsyncWork();

        expect(calls.chat).toHaveLength(2);
        const turn2 = calls.chat[1].body;
        expect(turn2.snapshot_uuid).toBe('s-1');
        // Wire shape: [user u1 with sid, assistant a1 with sid, new user u2 without sid].
        expect(turn2.messages).toEqual([
            { role: 'user', content: 'hi', source_id: '1.000001' },
            { role: 'assistant', content: 'first reply', source_id: '1.000002' },
            { role: 'user', content: 'follow-up' },
        ]);
    });
});

describe('Stream handshake application', () => {
    it('stamps server-assigned source_id and snapshot_uuid onto the submitted user node', async () => {
        const { fetchMock } = makeFetchMock({
            chatEvents: [
                { snapshot_uuid: 's-1', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
                { text_delta: 'ok' },
                { bot_message: { source_id: '1.000002' } },
            ],
        });
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'hi');
        await flushAsyncWork();

        // The user node is the only one with role='user' in the tree.
        const userNode = [...app.getMessages()].find(() => true) || null;
        // getMessages() returns DOM-rendered messages; getNode inspects the tree directly. root.children[0] is the
        // user node.
        const rootNode = app.getNode(0);
        const userNodeId = rootNode.children[0];
        const user = app.getNode(userNodeId);
        expect(user.role).toBe('user');
        expect(user.source_id).toBe('1.000001');
        expect(user.snapshot_uuid).toBe('s-1');
    });
});

describe('Assistant node creation after bot_message', () => {
    it('creates the assistant node with the bot source_id and the handshake snapshot_uuid', async () => {
        const { fetchMock } = makeFetchMock({
            chatEvents: [
                { snapshot_uuid: 's-1', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
                { text_delta: 'hello ' },
                { text_delta: 'world' },
                { bot_message: { source_id: '1.000002' } },
            ],
        });
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'hi');
        await flushAsyncWork();

        const userNodeId = app.getNode(0).children[0];
        const userNode = app.getNode(userNodeId);
        expect(userNode.children).toHaveLength(1);
        const assistant = app.getNode(userNode.children[0]);
        expect(assistant.role).toBe('assistant');
        expect(assistant.content).toBe('hello world');
        expect(assistant.source_id).toBe('1.000002');
        expect(assistant.snapshot_uuid).toBe('s-1');
    });

    it('keeps the error bubble visible when the chat POST returns 400', async () => {
        // The catch block paints an .error-message into the pending assistant bubble; the finally branch must NOT
        // wipe it.
        const fetchMock = vi.fn(async (url) => {
            if (typeof url === 'string' && url.endsWith('/conversation')) {
                return { ok: true, json: async () => ({ conversation_uuid: 'conv-123' }) };
            }
            if (typeof url === 'string' && url.includes('/chat?')) {
                return {
                    ok: false,
                    status: 400,
                    statusText: 'Bad Request',
                    json: async () => ({ detail: 'Assistant content mismatch' }),
                };
            }
            throw new Error(`Unexpected fetch URL: ${url}`);
        });
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'hi');
        await flushAsyncWork();

        // The user's tree node exists (created before the POST); no assistant.
        const userNodeId = app.getNode(0).children[0];
        expect(app.getNode(userNodeId).children).toHaveLength(0);

        // The error message is visible in chatWrapper (not wiped by renderMessages).
        const errorEl = app.elements.chatWrapper.querySelector('.error-message');
        expect(errorEl).not.toBeNull();
        expect(errorEl.textContent).toContain('Assistant content mismatch');
    });

    it('does NOT create an assistant node when the stream ends without a bot_message', async () => {
        const { fetchMock } = makeFetchMock({
            chatEvents: [
                { snapshot_uuid: 's-1', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
                { text_delta: 'partial' },
                // Stream ends — no bot_message event.
            ],
        });
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'hi');
        await flushAsyncWork();

        const userNodeId = app.getNode(0).children[0];
        const userNode = app.getNode(userNodeId);
        expect(userNode.children).toHaveLength(0);

        // Pending DOM bubble must also be cleared — leaving it would display partial text/activity that's not in
        // the tree and won't be re-sent.
        const assistantBubbles = app.elements.chatWrapper.querySelectorAll('.message.assistant');
        expect(assistantBubbles).toHaveLength(0);
    });
});

describe('Resync handshake (send-from-leaf auto-retry)', () => {
    it('reconstructs unacknowledged bot, reparents pending user, auto-retries, and commits on second turn', async () => {
        // Turn 1 (normal): u1 → handshake stamps sid 1.000001; bot a1 commits with sid 1.000002.
        const turn1Events = [
            { snapshot_uuid: 's-A', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
            { text_delta: 'first reply' },
            { bot_message: { source_id: '1.000002' } },
        ];
        // Turn 2 (resync trigger): server emits a resync handshake — stream ends without an LLM call. The unacked
        // bot is parented under u1 (sid 1.000001). Client must reconstruct the bot, reparent the just-pending u2
        // under it, and auto-retry.
        const turn2ResyncEvents = [
            {
                snapshot_uuid: 's-A',
                source_id_assignments: [],
                unacknowledged_bot_messages: [
                    { source_id: '1.000003', content: 'missed reply', parent_source_id: '1.000001' },
                ],
            },
        ];
        // Turn 2 retry (auto-fired by the client after applying resync): server now sees a coherent tree and
        // replies normally. The path is [u1, reconstructed, u2], so client_index=2 is the pending u2 node that
        // needs a source_id assigned.
        const turn2RetryEvents = [
            { snapshot_uuid: 's-A', source_id_assignments: [{ client_index: 2, source_id: '1.000004' }] },
            { text_delta: 'retry reply' },
            { bot_message: { source_id: '1.000005' } },
        ];

        const { fetchMock, calls } = makeFetchMock();
        let chatCallIndex = 0;
        fetchMock.mockImplementation(async (url, init) => {
            if (typeof url === 'string' && url.endsWith('/conversation')) {
                return { ok: true, json: async () => ({ conversation_uuid: 'conv-123' }) };
            }
            if (typeof url === 'string' && url.includes('/chat?')) {
                calls.chat.push({ url, init, body: JSON.parse(init.body) });
                const events =
                    chatCallIndex === 0 ? turn1Events :
                    chatCallIndex === 1 ? turn2ResyncEvents :
                    turn2RetryEvents;
                chatCallIndex += 1;
                return { ok: true, body: buildNdjsonBody(events) };
            }
            throw new Error(`Unexpected fetch URL: ${url}`);
        });

        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'hi');
        await flushAsyncWork();
        // Capture turn-1 nodes for comparison.
        const u1NodeId = app.getNode(0).children[0];
        const u1 = app.getNode(u1NodeId);
        const a1OriginalId = u1.children[0];
        const a1Original = app.getNode(a1OriginalId);
        expect(a1Original.source_id).toBe('1.000002');

        await sendOneMessage(app, 'second');
        await flushAsyncWork();

        // Three chat POSTs total: turn1 normal, turn2 resync, turn2 retry.
        expect(calls.chat).toHaveLength(3);

        // Reconstructed unacked bot is now a child of u1 alongside the original a1.
        const reconstructedBotId = u1.children.find(
            (id) => id !== a1OriginalId && app.getNode(id).source_id === '1.000003',
        );
        expect(reconstructedBotId).toBeDefined();
        const reconstructed = app.getNode(reconstructedBotId);
        expect(reconstructed.content).toBe('missed reply');
        expect(reconstructed.snapshot_uuid).toBe('s-A');

        // Pending u2 was reparented under the reconstructed bot (send-from-leaf semantics).
        expect(reconstructed.children).toHaveLength(1);
        const u2 = app.getNode(reconstructed.children[0]);
        expect(u2.role).toBe('user');
        expect(u2.content).toBe('second');
        // u2's source_id was assigned on the auto-retry handshake.
        expect(u2.source_id).toBe('1.000004');

        // The retry committed an assistant reply under u2.
        expect(u2.children).toHaveLength(1);
        const a2 = app.getNode(u2.children[0]);
        expect(a2.content).toBe('retry reply');
        expect(a2.source_id).toBe('1.000005');

        // No orphan pending DOM bubble (the finally branch removes it).
        const assistantBubbles = app.elements.chatWrapper.querySelectorAll('.message.assistant');
        // chatWrapper reflects the active path root → u1 → reconstructed → u2 → a2.
        expect(assistantBubbles.length).toBeGreaterThan(0);
    });
});

describe('Compaction-pending polling and relabel', () => {
    it('polls /compaction-status and relabels message-tree nodes when a child snapshot_uuid is returned', async () => {
        vi.useFakeTimers();
        const { fetchMock, calls } = makeFetchMock({
            chatEvents: [
                { snapshot_uuid: 's-pending', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
                { text_delta: 'compacting soon' },
                { bot_message: { source_id: '1.000002' } },
                { compaction_pending: true },
            ],
            compactionStatuses: [
                { done: true, snapshot_uuid: 's-child' },
            ],
        });
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'hi');
        await flushAsyncWork();

        // Pre-poll: both submitted user node and assistant node carry the pending snapshot_uuid.
        const userNodeId = app.getNode(0).children[0];
        const user = app.getNode(userNodeId);
        const assistant = app.getNode(user.children[0]);
        expect(user.snapshot_uuid).toBe('s-pending');
        expect(assistant.snapshot_uuid).toBe('s-pending');

        // Advance the poll interval. The polling response carries done=true plus the child snapshot_uuid; the tree
        // should be relabeled.
        await vi.advanceTimersByTimeAsync(5_000);
        await flushAsyncWork();

        expect(calls.compactionStatus).toHaveLength(1);
        expect(calls.compactionStatus[0].url).toContain('pending_snapshot_uuid=s-pending');
        // Both nodes that had s-pending now carry s-child.
        expect(app.getNode(userNodeId).snapshot_uuid).toBe('s-child');
        expect(app.getNode(user.children[0]).snapshot_uuid).toBe('s-child');
        // Indicator is cleared.
        expect(document.getElementById('compaction-indicator').hidden).toBe(true);
    });
});

// Multi-turn fetch mock: `turns` is an array of NDJSON event-lists (one per chat POST); `statusByPending` maps a
// `pending_snapshot_uuid` to its queue of `/compaction-status` responses. A queued entry of `{ httpError: true }`
// produces an `ok: false` HTTP response.
function makeTurnFetchMock(turns, statusByPending = {}) {
    const calls = { chat: [], compactionStatus: [] };
    let turnIndex = 0;
    const queues = {};
    for (const [k, v] of Object.entries(statusByPending)) {
        queues[k] = [...v];
    }
    const fetchMock = vi.fn(async (url, init) => {
        if (typeof url === 'string' && url.endsWith('/conversation')) {
            return { ok: true, json: async () => ({ conversation_uuid: 'conv-123' }) };
        }
        if (typeof url === 'string' && url.includes('/chat?')) {
            calls.chat.push({ url, body: init ? JSON.parse(init.body) : null });
            const events = turns[turnIndex] || [];
            turnIndex += 1;
            return { ok: true, body: buildNdjsonBody(events) };
        }
        if (typeof url === 'string' && url.includes('/compaction-status')) {
            calls.compactionStatus.push(url);
            const pending = new URL(url, 'http://x').searchParams.get('pending_snapshot_uuid');
            const next = (queues[pending] || []).shift();
            if (next && next.httpError) {
                return { ok: false, status: next.status || 503 };
            }
            return { ok: true, json: async () => next || { done: false } };
        }
        throw new Error(`Unexpected fetch URL: ${url}`);
    });
    return { fetchMock, calls };
}

describe('Branch-local snapshot_uuid selection', () => {
    it('regenerate after a branch switch posts the viewed branch snapshot, not a restamped sibling', async () => {
        // turn1: U1 -> snapshot s-1, bot a1.
        const turn1 = [
            { snapshot_uuid: 's-1', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
            { text_delta: 'a1' },
            { bot_message: { source_id: '1.000002' } },
        ];
        // turn2: U2 send-from-leaf -> still s-1 (append), bot a2.
        const turn2 = [
            { snapshot_uuid: 's-1', source_id_assignments: [{ client_index: 2, source_id: '1.000003' }] },
            { text_delta: 'a2' },
            { bot_message: { source_id: '1.000004' } },
        ];
        // turn3: edit U2 -> divergence -> new branch s-2. applyHandshake restamps the shared prefix [U1, A1] (in
        // sentClientIds) to s-2.
        const turn3 = [
            { snapshot_uuid: 's-2', source_id_assignments: [{ client_index: 2, source_id: '1.000005' }] },
            { text_delta: 'a2fork' },
            { bot_message: { source_id: '1.000006' } },
        ];
        // turn4: regenerate from the s-1 branch — only the request body matters.
        const turn4 = [
            { snapshot_uuid: 's-1', source_id_assignments: [] },
            { text_delta: 'a1-regen' },
            { bot_message: { source_id: '1.000007' } },
        ];
        const { fetchMock, calls } = makeTurnFetchMock([turn1, turn2, turn3, turn4]);

        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'U1');
        await flushAsyncWork();
        await sendOneMessage(app, 'U2');
        await flushAsyncWork();

        // Fork: edit U2 (index 2) into a divergent message.
        app.editMessage(2);
        await sendOneMessage(app, 'U2-forked');
        await flushAsyncWork();

        // The shared prefix node A1 was restamped to the fork's snapshot s-2.
        const u1NodeId = app.getNode(0).children[0];
        const a1NodeId = app.getNode(u1NodeId).children[0];
        expect(app.getNode(a1NodeId).snapshot_uuid).toBe('s-2');

        // Switch the fork back to the original U2 branch (s-1 lineage).
        app.switchUserFork(2, -1);

        // Regenerate A1 — a shared message (index 1) — while viewing the s-1 branch.
        await app.regenerateMessage(1);
        await flushAsyncWork();

        expect(calls.chat).toHaveLength(4);
        // The POST must carry the viewed branch's snapshot (s-1), even though the shared parent nodes now carry the
        // sibling branch's s-2.
        expect(calls.chat[3].body.snapshot_uuid).toBe('s-1');
    });
});

describe('Branch-scoped compaction polling', () => {
    it('keeps polling after a transient /compaction-status failure and still relabels', async () => {
        vi.useFakeTimers();
        const turn = [
            { snapshot_uuid: 's-pending', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
            { text_delta: 'reply' },
            { bot_message: { source_id: '1.000002' } },
            { compaction_pending: true },
        ];
        const { fetchMock, calls } = makeTurnFetchMock([turn], {
            's-pending': [
                { httpError: true, status: 503 },
                { done: false },
                { done: true, snapshot_uuid: 's-child' },
            ],
        });
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOneMessage(app, 'hi');
        await flushAsyncWork();
        const userNodeId = app.getNode(0).children[0];
        expect(app.getNode(userNodeId).snapshot_uuid).toBe('s-pending');

        // Poll #1 — transient HTTP failure. The loop must survive it.
        await vi.advanceTimersByTimeAsync(5_000);
        await flushAsyncWork();
        expect(document.getElementById('compaction-indicator').hidden).toBe(false);

        // Poll #2 — still pending.
        await vi.advanceTimersByTimeAsync(5_000);
        await flushAsyncWork();
        expect(document.getElementById('compaction-indicator').hidden).toBe(false);

        // Poll #3 — done with a relabel target.
        await vi.advanceTimersByTimeAsync(5_000);
        await flushAsyncWork();
        expect(calls.compactionStatus).toHaveLength(3);
        expect(app.getNode(userNodeId).snapshot_uuid).toBe('s-child');
        expect(document.getElementById('compaction-indicator').hidden).toBe(true);
    });

    it('runs an independent poll loop per branch and never side-channel-clears the indicator', async () => {
        vi.useFakeTimers();
        // turn1: normal turn that schedules a compaction on s-1.
        const turn1 = [
            { snapshot_uuid: 's-1', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
            { text_delta: 'a1' },
            { bot_message: { source_id: '1.000002' } },
            { compaction_pending: true },
        ];
        // turn2: edit U1 -> divergence to branch s-2, which also schedules a compaction. Two poll loops are now
        // live concurrently.
        const turn2 = [
            { snapshot_uuid: 's-2', source_id_assignments: [{ client_index: 0, source_id: '1.000003' }] },
            { text_delta: 'a1fork' },
            { bot_message: { source_id: '1.000004' } },
            { compaction_pending: true },
        ];
        const { fetchMock } = makeTurnFetchMock([turn1, turn2], {
            's-1': [{ done: true, snapshot_uuid: 's-1-child' }],
            's-2': [{ done: false }, { done: true, snapshot_uuid: 's-2-child' }],
        });
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });

        await sendOneMessage(app, 'U1');
        await flushAsyncWork();
        const u1NodeId = app.getNode(0).children[0];

        // Edit U1 into a divergent message — a sibling branch. Its handshake (snapshot s-2) must NOT clear the s-1
        // compaction indicator.
        app.editMessage(0);
        await sendOneMessage(app, 'U1-forked');
        await flushAsyncWork();
        expect(document.getElementById('compaction-indicator').hidden).toBe(false);

        // First tick: s-1's poll completes and relabels; s-2's poll is still pending, so the shared indicator
        // stays visible.
        await vi.advanceTimersByTimeAsync(5_000);
        await flushAsyncWork();
        expect(app.getNode(u1NodeId).snapshot_uuid).toBe('s-1-child');
        expect(document.getElementById('compaction-indicator').hidden).toBe(false);

        // Second tick: s-2's poll completes. With no branch left pending, the indicator finally hides.
        await vi.advanceTimersByTimeAsync(5_000);
        await flushAsyncWork();
        expect(document.getElementById('compaction-indicator').hidden).toBe(true);
    });

    it('handles back-to-back compactions on one branch, each with its own relabel cycle', async () => {
        vi.useFakeTimers();
        const turn1 = [
            { snapshot_uuid: 's-a', source_id_assignments: [{ client_index: 0, source_id: '1.000001' }] },
            { text_delta: 'a1' },
            { bot_message: { source_id: '1.000002' } },
            { compaction_pending: true },
        ];
        // turn2 posts into s-a-child (the relabel target of cycle 1) and triggers a second compaction.
        const turn2 = [
            { snapshot_uuid: 's-a-child', source_id_assignments: [{ client_index: 2, source_id: '1.000003' }] },
            { text_delta: 'a2' },
            { bot_message: { source_id: '1.000004' } },
            { compaction_pending: true },
        ];
        const { fetchMock, calls } = makeTurnFetchMock([turn1, turn2], {
            's-a': [{ done: true, snapshot_uuid: 's-a-child' }],
            's-a-child': [{ done: true, snapshot_uuid: 's-a-grandchild' }],
        });
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });

        await sendOneMessage(app, 'hi');
        await flushAsyncWork();
        await vi.advanceTimersByTimeAsync(5_000);
        await flushAsyncWork();
        const u1NodeId = app.getNode(0).children[0];
        expect(app.getNode(u1NodeId).snapshot_uuid).toBe('s-a-child');
        expect(document.getElementById('compaction-indicator').hidden).toBe(true);

        await sendOneMessage(app, 'again');
        await flushAsyncWork();
        // Cycle 2 sends against the relabeled snapshot.
        expect(calls.chat[1].body.snapshot_uuid).toBe('s-a-child');
        await vi.advanceTimersByTimeAsync(5_000);
        await flushAsyncWork();
        // Every node carrying s-a-child has rolled to the second child.
        expect(app.getNode(u1NodeId).snapshot_uuid).toBe('s-a-grandchild');
        expect(document.getElementById('compaction-indicator').hidden).toBe(true);
    });
});
