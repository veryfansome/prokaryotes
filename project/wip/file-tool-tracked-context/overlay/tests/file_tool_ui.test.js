/**
 * Vitest coverage for the new `file_tool` formatter in `formatToolCallMarkdown`.
 *
 * The formatter itself is closed over inside `createChatApp`, so this test exercises it
 * via the same path the runtime uses: pushing a `tool_call` event and inspecting the
 * `.message-activity-tool_call` rendering.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createChatApp } from '../scripts/static/ui.js';

function renderBaseDOM() {
    document.body.innerHTML = `
        <div id="chatContainer"><div id="chatWrapper"></div></div>
        <textarea id="chatInput"></textarea>
        <button id="sendButton" disabled>Send</button>
        <div id="editStatus" hidden></div>
    `;
}

function createControllableFetchMock() {
    const encoder = new TextEncoder();
    let controller = null;

    const body = new ReadableStream({
        start(streamController) {
            controller = streamController;
        },
    });

    const fetchMock = vi.fn(async url => {
        if (url.endsWith('/conversation')) {
            return { ok: true, json: async () => ({ conversation_uuid: 'conv-123' }) };
        }
        if (url.includes('/chat?')) {
            return { ok: true, body };
        }
        throw new Error(`Unexpected URL: ${url}`);
    });

    const waitForReady = async () => {
        while (!controller) {
            await Promise.resolve();
        }
    };

    return {
        fetchMock,
        waitForReady,
        pushToolCall(name, args = '{}') {
            const payload = JSON.stringify({ tool_call: { name, arguments: args } });
            controller.enqueue(encoder.encode(`${payload}\n`));
        },
        close() {
            controller.close();
        },
    };
}

async function flushAsyncWork() {
    await Promise.resolve();
    await Promise.resolve();
}

async function pushFileToolCallAndCapture(args) {
    const controlledFetch = createControllableFetchMock();
    const app = createChatApp({
        doc: document,
        fetchImpl: controlledFetch.fetchMock,
        navigatorImpl: {},
    });
    app.elements.chatInput.value = 'hi';
    app.elements.chatInput.dispatchEvent(new Event('input'));
    const sendPromise = app.sendMessage();
    await controlledFetch.waitForReady();

    controlledFetch.pushToolCall('file_tool', JSON.stringify(args));
    await flushAsyncWork();
    const toolCallEl = app.elements.chatWrapper.querySelector('.message-activity-tool_call');
    controlledFetch.close();
    await sendPromise.catch(() => undefined);
    return toolCallEl;
}

describe('file_tool tool call rendering', () => {
    beforeEach(() => {
        renderBaseDOM();
    });

    it('renders read with start_line', async () => {
        const el = await pushFileToolCallAndCapture({
            action: 'read',
            path: '/app/foo.py',
            expected_revision: null,
            start_line: 10,
            end_line: null,
            new_text: null,
        });
        expect(el).not.toBeNull();
        expect(el.textContent).toContain('file_tool');
        expect(el.textContent).toContain('Reading');
        expect(el.textContent).toContain('/app/foo.py');
        expect(el.textContent).toContain('from line 10');
    });

    it('renders read without start_line', async () => {
        const el = await pushFileToolCallAndCapture({
            action: 'read',
            path: '/app/foo.py',
            expected_revision: null,
            start_line: null,
            end_line: null,
            new_text: null,
        });
        expect(el.textContent).toContain('Reading');
        expect(el.textContent).toContain('/app/foo.py');
        expect(el.textContent).not.toContain('from line');
    });

    it('renders replace_lines with new_text fenced', async () => {
        const el = await pushFileToolCallAndCapture({
            action: 'replace_lines',
            path: '/app/foo.py',
            expected_revision: 'abc',
            start_line: 5,
            end_line: 7,
            new_text: 'updated\ncontent',
        });
        expect(el.textContent).toContain('Editing');
        expect(el.textContent).toContain('lines 5');
        expect(el.textContent).toContain('updated');
        expect(el.textContent).toContain('content');
    });

    it('renders insert_lines with new_text fenced', async () => {
        const el = await pushFileToolCallAndCapture({
            action: 'insert_lines',
            path: '/app/foo.py',
            expected_revision: 'abc',
            start_line: 5,
            end_line: null,
            new_text: 'inserted',
        });
        expect(el.textContent).toContain('Inserting');
        expect(el.textContent).toContain('line 5');
        expect(el.textContent).toContain('inserted');
    });

    it('renders delete_lines without new_text', async () => {
        const el = await pushFileToolCallAndCapture({
            action: 'delete_lines',
            path: '/app/foo.py',
            expected_revision: 'abc',
            start_line: 5,
            end_line: 9,
            new_text: null,
        });
        expect(el.textContent).toContain('Deleting');
        expect(el.textContent).toContain('lines 5');
    });

    it('does not render the null-valued strict-mode parameters', async () => {
        const el = await pushFileToolCallAndCapture({
            action: 'read',
            path: '/app/foo.py',
            expected_revision: null,
            start_line: null,
            end_line: null,
            new_text: null,
        });
        expect(el.textContent).not.toContain('null');
        expect(el.textContent).not.toContain('expected_revision');
        expect(el.textContent).not.toContain('end_line');
    });
});
