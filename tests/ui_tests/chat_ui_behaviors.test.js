/**
 * DOM/fetch tests for chat UI behaviors that aren't part of the wire protocol:
 * edit-mode entry/cancel, fork switching across user-message branches, and code-block copy buttons.
 *
 * Protocol tests live in `chat_protocol_integration.test.js`; pure parser tests live in `ui.test.js` and
 * `conversation_client.test.js`.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createChatApp } from '../../scripts/static/ui.js';

function renderBaseDOM() {
    document.body.innerHTML = `
        <div id="chatContainer"><div id="chatWrapper"></div></div>
        <textarea id="chatInput"></textarea>
        <button id="sendButton" disabled>Send</button>
        <div id="editStatus" hidden></div>
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

// Stub a sequence of chat turns. Each entry is the assistant text for that turn; source_ids are auto-assigned so
// each turn lands as a coherent user/bot pair.
function makeSequencedFetchMock(turns) {
    let turnIndex = 0;
    let nextSourceId = 1;
    return vi.fn(async (url, init) => {
        if (typeof url === 'string' && url.endsWith('/conversation')) {
            return { ok: true, json: async () => ({ conversation_uuid: 'conv-123' }) };
        }
        if (typeof url === 'string' && url.includes('/chat?')) {
            const text = turns[Math.min(turnIndex, turns.length - 1)];
            const body = JSON.parse(init.body);
            const userClientIdx = body.messages.length - 1;
            // Stamp one source_id for the just-sent user node, one for the bot reply.
            const userSid = `1.${String(nextSourceId++).padStart(6, '0')}`;
            const botSid = `1.${String(nextSourceId++).padStart(6, '0')}`;
            turnIndex += 1;
            return {
                ok: true,
                body: buildNdjsonBody([
                    {
                        snapshot_uuid: 's-A',
                        source_id_assignments: [{ client_index: userClientIdx, source_id: userSid }],
                    },
                    { text_delta: text },
                    { bot_message: { source_id: botSid } },
                ]),
            };
        }
        throw new Error(`Unexpected URL: ${url}`);
    });
}

async function flushAsyncWork() {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
}

async function sendOne(app, text) {
    app.elements.chatInput.value = text;
    app.elements.chatInput.dispatchEvent(new Event('input'));
    await app.sendMessage();
    await flushAsyncWork();
}

function asRoleContent(messages) {
    return messages.map((m) => ({ role: m.role, content: m.content }));
}

beforeEach(() => {
    renderBaseDOM();
});

describe('Edit mode', () => {
    it('enters edit mode when clicking a previous user message', async () => {
        const fetchMock = makeSequencedFetchMock(['A1', 'A2']);
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOne(app, 'U1');
        await sendOne(app, 'U2');

        const editable = app.elements.chatWrapper.querySelector(
            '.message.user .message-content[data-message-index="2"]',
        );
        expect(editable).not.toBeNull();
        editable.dispatchEvent(new MouseEvent('click', { bubbles: true }));

        expect(app.getIsEditing()).toBe(true);
        expect(app.elements.editStatus.hidden).toBe(false);
        expect(app.elements.chatWrapper.querySelector('.editing-source')).not.toBeNull();
    });

    it('cancels edit mode with Escape and restores the active branch', async () => {
        const fetchMock = makeSequencedFetchMock(['A1', 'A2']);
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOne(app, 'U1');
        await sendOne(app, 'U2');

        app.editMessage(2);
        expect(app.getIsEditing()).toBe(true);
        expect(app.elements.editStatus.hidden).toBe(false);

        app.handleKeyDown(new KeyboardEvent('keydown', { key: 'Escape' }));

        expect(app.getIsEditing()).toBe(false);
        expect(app.elements.editStatus.hidden).toBe(true);
        // The active branch is intact — all four messages present.
        expect(asRoleContent(app.getMessages())).toEqual([
            { role: 'user', content: 'U1' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'U2' },
            { role: 'assistant', content: 'A2' },
        ]);
    });

    it('cancels edit mode on outside mousedown', async () => {
        const fetchMock = makeSequencedFetchMock(['A1', 'A2']);
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOne(app, 'U1');
        await sendOne(app, 'U2');

        app.editMessage(2);
        expect(app.getIsEditing()).toBe(true);

        document.body.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));

        expect(app.getIsEditing()).toBe(false);
        expect(app.elements.editStatus.hidden).toBe(true);
    });
});

describe('Fork switching', () => {
    it('forks the user message and lets the user switch between branches', async () => {
        const fetchMock = makeSequencedFetchMock(['A1', 'A2', 'AFork']);
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOne(app, 'U1');
        await sendOne(app, 'U2');

        // Fork: edit u2's slot, send a different text.
        app.editMessage(2);
        await sendOne(app, 'U2 - forked');

        expect(asRoleContent(app.getMessages())).toEqual([
            { role: 'user', content: 'U1' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'U2 - forked' },
            { role: 'assistant', content: 'AFork' },
        ]);

        // The forked user node renders a 2/2 fork indicator.
        const indicator = app.elements.chatWrapper.querySelector('.message.user .fork-indicator');
        expect(indicator).not.toBeNull();
        expect(indicator.textContent).toBe('2/2');

        // Clicking the previous-fork button switches back to the original branch.
        const prevBtn = app.elements.chatWrapper.querySelector('.message.user .fork-nav-btn');
        expect(prevBtn).not.toBeNull();
        prevBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));

        expect(asRoleContent(app.getMessages())).toEqual([
            { role: 'user', content: 'U1' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'U2' },
            { role: 'assistant', content: 'A2' },
        ]);
        const switched = app.elements.chatWrapper.querySelector('.message.user .fork-indicator');
        expect(switched.textContent).toBe('1/2');
    });
});

describe('Copy buttons', () => {
    it('adds a copy button to each code block in assistant messages', async () => {
        const fetchMock = makeSequencedFetchMock(['```js\nconsole.log("hi")\n```']);
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOne(app, 'Show me a one-liner.');

        const copyBtn = app.elements.chatWrapper.querySelector('.message.assistant .copy-btn');
        expect(copyBtn).not.toBeNull();
    });

    it('writes code content to clipboard when the copy button is clicked', async () => {
        // The copy handler calls `navigator.clipboard.writeText` directly (not the injected `navigatorImpl`), so
        // we stub the global.
        const writeText = vi.fn().mockResolvedValue(undefined);
        Object.defineProperty(navigator, 'clipboard', {
            value: { writeText },
            configurable: true,
        });
        const fetchMock = makeSequencedFetchMock(['```js\nconsole.log("hi")\n```']);
        const app = createChatApp({ doc: document, fetchImpl: fetchMock, navigatorImpl: {} });
        await sendOne(app, 'Show me a one-liner.');

        const copyBtn = app.elements.chatWrapper.querySelector('.message.assistant .copy-btn');
        copyBtn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        await flushAsyncWork();

        expect(writeText).toHaveBeenCalledTimes(1);
        expect(writeText.mock.calls[0][0]).toContain('console.log("hi")');
    });
});
