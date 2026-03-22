import { beforeEach, describe, expect, it, vi } from 'vitest';

import { buildChatQueryParams, createChatApp, parseStreamPayloadLine } from '../scripts/static/ui.js';

function renderBaseDOM() {
    document.body.innerHTML = `
        <div id="chatContainer"><div id="chatWrapper"></div></div>
        <textarea id="chatInput"></textarea>
        <button id="sendButton" disabled>Send</button>
        <div id="editStatus" hidden></div>
    `;
}

function streamFromJsonLines(lines) {
    const encoder = new TextEncoder();
    return new ReadableStream({
        start(controller) {
            for (const line of lines) {
                controller.enqueue(encoder.encode(`${line}\n`));
            }
            controller.close();
        },
    });
}

function createFetchMock(chatPayloads) {
    let chatCall = 0;

    return vi.fn(async (url) => {
        if (url.endsWith('/conversation')) {
            return {
                ok: true,
                json: async () => ({ conversation_uuid: 'conv-123' }),
            };
        }

        if (url.includes('/chat?')) {
            const payload = chatPayloads[Math.min(chatCall, chatPayloads.length - 1)];
            chatCall += 1;
            return {
                ok: true,
                body: streamFromJsonLines(payload),
            };
        }

        throw new Error(`Unexpected URL: ${url}`);
    });
}

function createControllableFetchMock() {
    const encoder = new TextEncoder();
    let controller = null;

    const body = new ReadableStream({
        start(streamController) {
            controller = streamController;
        },
    });

    const fetchMock = vi.fn(async (url) => {
        if (url.endsWith('/conversation')) {
            return {
                ok: true,
                json: async () => ({ conversation_uuid: 'conv-123' }),
            };
        }

        if (url.includes('/chat?')) {
            return {
                ok: true,
                body,
            };
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
        pushTextDelta(text) {
            controller.enqueue(encoder.encode(`${JSON.stringify({ text_delta: text })}\n`));
        },
        close() {
            controller.close();
        },
    };
}

function mockScrollableContainer(
    container,
    { scrollHeight = 1600, clientHeight = 600, initialScrollTop = 1000 } = {},
) {
    let currentScrollTop = initialScrollTop;
    let currentScrollHeight = scrollHeight;

    Object.defineProperty(container, 'clientHeight', {
        configurable: true,
        get: () => clientHeight,
    });
    Object.defineProperty(container, 'scrollHeight', {
        configurable: true,
        get: () => currentScrollHeight,
    });
    Object.defineProperty(container, 'scrollTop', {
        configurable: true,
        get: () => currentScrollTop,
        set: value => {
            currentScrollTop = value;
        },
    });

    return {
        getMaxScrollTop: () => Math.max(0, currentScrollHeight - clientHeight),
        getScrollTop: () => currentScrollTop,
        setScrollTop: value => {
            currentScrollTop = value;
        },
        setScrollHeight: value => {
            currentScrollHeight = value;
        },
    };
}

async function flushAsyncWork() {
    await Promise.resolve();
    await Promise.resolve();
}

function asRoleContent(messages) {
    return messages.map(message => ({ role: message.role, content: message.content }));
}

describe('ui.js helpers', () => {
    it('parses text_delta stream payload', () => {
        expect(parseStreamPayloadLine('{"text_delta":"hello"}')).toEqual({ type: 'text_delta', text_delta: 'hello' });
    });

    it('throws on invalid payload json', () => {
        expect(() => parseStreamPayloadLine('{bad json}')).toThrow('Invalid stream payload');
    });

    it('builds query string with optional geolocation', () => {
        expect(buildChatQueryParams('America/Los_Angeles', null, null)).toBe('time_zone=America%2FLos_Angeles');
        expect(buildChatQueryParams('America/Los_Angeles', 1.23, 4.56)).toContain('latitude=1.23');
        expect(buildChatQueryParams('America/Los_Angeles', 1.23, 4.56)).toContain('longitude=4.56');
    });
});

describe('createChatApp messageTree flow', () => {
    beforeEach(() => {
        renderBaseDOM();
    });

    it('sends a message and streams assistant response', async () => {
        const fetchMock = createFetchMock([[JSON.stringify({ text_delta: 'Hel' }), JSON.stringify({ text_delta: 'lo' })]]);

        const app = createChatApp({
            doc: document,
            fetchImpl: fetchMock,
            navigatorImpl: {},
        });

        app.elements.chatInput.value = 'Hi';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        expect(asRoleContent(app.getMessages())).toEqual([
            { role: 'user', content: 'Hi' },
            { role: 'assistant', content: 'Hello' },
        ]);
    });

    it('keeps scrolling to the textarea while assistant tokens stream', async () => {
        const fetchMock = createFetchMock([[JSON.stringify({ text_delta: 'Hel' }), JSON.stringify({ text_delta: 'lo' })]]);

        const app = createChatApp({
            doc: document,
            fetchImpl: fetchMock,
            navigatorImpl: {},
        });

        const scrollIntoViewSpy = vi.fn();
        app.elements.chatInput.scrollIntoView = scrollIntoViewSpy;

        app.elements.chatInput.value = 'Hi';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        expect(scrollIntoViewSpy).toHaveBeenCalled();
    });

    it('scrolls the chat container to its lowest possible offset', async () => {
        const fetchMock = createFetchMock([[JSON.stringify({ text_delta: 'Hello' })]]);

        const app = createChatApp({
            doc: document,
            fetchImpl: fetchMock,
            navigatorImpl: {},
        });

        const scrollState = mockScrollableContainer(app.elements.chatContainer, {
            clientHeight: 600,
            initialScrollTop: 0,
            scrollHeight: 1600,
        });

        app.elements.chatInput.scrollIntoView = vi.fn();
        app.elements.chatInput.value = 'Hi';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        expect(scrollState.getScrollTop()).toBe(scrollState.getMaxScrollTop());
    });

    it('pauses auto-scroll while generating when user scrolls up, then resumes at bottom', async () => {
        const controlledFetch = createControllableFetchMock();
        const app = createChatApp({
            doc: document,
            fetchImpl: controlledFetch.fetchMock,
            navigatorImpl: {},
        });

        const scrollIntoViewSpy = vi.fn();
        app.elements.chatInput.scrollIntoView = scrollIntoViewSpy;
        const scrollState = mockScrollableContainer(app.elements.chatContainer);

        app.elements.chatInput.value = 'Hi';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        const sendPromise = app.sendMessage();
        await controlledFetch.waitForReady();

        controlledFetch.pushTextDelta('first');
        await flushAsyncWork();
        const callsBeforePause = scrollIntoViewSpy.mock.calls.length;

        scrollState.setScrollTop(200);
        app.elements.chatContainer.dispatchEvent(new Event('scroll'));
        await flushAsyncWork();

        controlledFetch.pushTextDelta('second');
        await flushAsyncWork();
        expect(scrollIntoViewSpy.mock.calls.length).toBe(callsBeforePause);

        scrollState.setScrollTop(scrollState.getMaxScrollTop());
        app.elements.chatContainer.dispatchEvent(new Event('scroll'));
        await flushAsyncWork();

        controlledFetch.pushTextDelta('third');
        controlledFetch.close();
        await sendPromise;

        expect(scrollIntoViewSpy.mock.calls.length).toBeGreaterThan(callsBeforePause);
    });

    it('resumes auto-scroll while generating when user starts typing', async () => {
        const controlledFetch = createControllableFetchMock();
        const app = createChatApp({
            doc: document,
            fetchImpl: controlledFetch.fetchMock,
            navigatorImpl: {},
        });

        const scrollIntoViewSpy = vi.fn();
        app.elements.chatInput.scrollIntoView = scrollIntoViewSpy;
        const scrollState = mockScrollableContainer(app.elements.chatContainer);

        app.elements.chatInput.value = 'Hi';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        const sendPromise = app.sendMessage();
        await controlledFetch.waitForReady();

        controlledFetch.pushTextDelta('first');
        await flushAsyncWork();

        scrollState.setScrollTop(200);
        app.elements.chatContainer.dispatchEvent(new Event('scroll'));
        await flushAsyncWork();

        controlledFetch.pushTextDelta('second');
        await flushAsyncWork();
        const callsWhilePaused = scrollIntoViewSpy.mock.calls.length;

        app.elements.chatInput.value = 'draft';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await flushAsyncWork();
        expect(scrollIntoViewSpy.mock.calls.length).toBeGreaterThan(callsWhilePaused);

        const callsAfterTyping = scrollIntoViewSpy.mock.calls.length;
        controlledFetch.pushTextDelta('third');
        controlledFetch.close();
        await sendPromise;

        expect(scrollIntoViewSpy.mock.calls.length).toBeGreaterThan(callsAfterTyping);
    });

    it('keeps textarea bottom visible while typing multiline input', () => {
        const fetchMock = createFetchMock([[JSON.stringify({ text_delta: 'A1' })]]);
        const app = createChatApp({
            doc: document,
            fetchImpl: fetchMock,
            navigatorImpl: {},
        });

        const scrollIntoViewSpy = vi.fn();
        app.elements.chatInput.scrollIntoView = scrollIntoViewSpy;
        app.elements.chatInput.getBoundingClientRect = vi.fn(() => ({
            bottom: 900,
            height: 120,
            left: 0,
            right: 0,
            top: 780,
            width: 0,
            x: 0,
            y: 780,
            toJSON() {
                return {};
            },
        }));

        app.elements.chatInput.focus();
        app.elements.chatInput.value = 'line1\nline2';
        app.elements.chatInput.dispatchEvent(new Event('input'));

        expect(scrollIntoViewSpy).toHaveBeenCalled();
    });

    it('scrolls textarea into view when editing a previous user message by click', async () => {
        const fetchMock = createFetchMock([
            [JSON.stringify({ text_delta: 'A1' })],
            [JSON.stringify({ text_delta: 'A2' })],
        ]);

        const app = createChatApp({
            doc: document,
            fetchImpl: fetchMock,
            navigatorImpl: {},
        });

        app.elements.chatInput.value = 'U1';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        app.elements.chatInput.value = 'U2';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        const scrollIntoViewSpy = vi.fn();
        app.elements.chatInput.scrollIntoView = scrollIntoViewSpy;

        const editableMessage = app.elements.chatWrapper.querySelector(
            '.message.user .message-content[data-message-index="2"]',
        );
        expect(editableMessage).not.toBeNull();
        editableMessage.dispatchEvent(new MouseEvent('click', { bubbles: true }));

        expect(app.getIsEditing()).toBe(true);
        expect(scrollIntoViewSpy).toHaveBeenCalled();
    });

    it('cancels edit mode with Escape and restores active branch', async () => {
        const fetchMock = createFetchMock([
            [JSON.stringify({ text_delta: 'A1' })],
            [JSON.stringify({ text_delta: 'A2' })],
        ]);

        const app = createChatApp({
            doc: document,
            fetchImpl: fetchMock,
            navigatorImpl: {},
        });

        app.elements.chatInput.value = 'U1';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        app.elements.chatInput.value = 'U2';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        app.editMessage(2);
        expect(app.getIsEditing()).toBe(true);
        expect(asRoleContent(app.getMessages())).toEqual([
            { role: 'user', content: 'U1' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'U2' },
        ]);
        expect(app.elements.editStatus.hidden).toBe(false);
        expect(app.elements.chatWrapper.querySelector('.editing-source')).not.toBeNull();

        app.handleKeyDown(new KeyboardEvent('keydown', { key: 'Escape' }));

        expect(app.getIsEditing()).toBe(false);
        expect(app.elements.editStatus.hidden).toBe(true);
        expect(asRoleContent(app.getMessages())).toEqual([
            { role: 'user', content: 'U1' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'U2' },
            { role: 'assistant', content: 'A2' },
        ]);
    });

    it('shows fork buttons and switches between user forks', async () => {
        const fetchMock = createFetchMock([
            [JSON.stringify({ text_delta: 'A1' })],
            [JSON.stringify({ text_delta: 'A2' })],
            [JSON.stringify({ text_delta: 'AFork' })],
        ]);

        const app = createChatApp({
            doc: document,
            fetchImpl: fetchMock,
            navigatorImpl: {},
        });

        app.elements.chatInput.value = 'U1';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        app.elements.chatInput.value = 'U2';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        app.editMessage(2);
        app.elements.chatInput.value = 'U2 - forked';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        expect(asRoleContent(app.getMessages())).toEqual([
            { role: 'user', content: 'U1' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'U2 - forked' },
            { role: 'assistant', content: 'AFork' },
        ]);

        const indicator = app.elements.chatWrapper.querySelector('.message.user .fork-indicator');
        expect(indicator).not.toBeNull();
        expect(indicator.textContent).toBe('2/2');

        const previousForkButton = app.elements.chatWrapper.querySelector('.message.user .fork-nav-btn');
        expect(previousForkButton).not.toBeNull();
        previousForkButton.dispatchEvent(new MouseEvent('click', { bubbles: true }));

        expect(asRoleContent(app.getMessages())).toEqual([
            { role: 'user', content: 'U1' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'U2' },
            { role: 'assistant', content: 'A2' },
        ]);

        const switchedIndicator = app.elements.chatWrapper.querySelector('.message.user .fork-indicator');
        expect(switchedIndicator.textContent).toBe('1/2');
    });

    it('cancels edit mode on outside mousedown', async () => {
        const fetchMock = createFetchMock([
            [JSON.stringify({ text_delta: 'A1' })],
            [JSON.stringify({ text_delta: 'A2' })],
        ]);

        const app = createChatApp({
            doc: document,
            fetchImpl: fetchMock,
            navigatorImpl: {},
        });

        app.elements.chatInput.value = 'U1';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        app.elements.chatInput.value = 'U2';
        app.elements.chatInput.dispatchEvent(new Event('input'));
        await app.sendMessage();

        app.editMessage(2);
        expect(app.getIsEditing()).toBe(true);
        expect(app.elements.editStatus.hidden).toBe(false);
        expect(app.elements.chatWrapper.querySelector('.editing-source')).not.toBeNull();

        document.body.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));

        expect(app.getIsEditing()).toBe(false);
        expect(app.elements.editStatus.hidden).toBe(true);
        expect(asRoleContent(app.getMessages())).toEqual([
            { role: 'user', content: 'U1' },
            { role: 'assistant', content: 'A1' },
            { role: 'user', content: 'U2' },
            { role: 'assistant', content: 'A2' },
        ]);
    });
});
