import { beforeEach, describe, expect, it, vi } from 'vitest';

import { buildChatQueryParams, createChatApp, parseStreamPayloadLine } from '../scripts/static/ui.js';

// 1) DOM fixtures + network stream mocks
function renderBaseDOM() {
    document.body.innerHTML = `
        <div id="chatContainer"><div id="chatWrapper"></div></div>
        <textarea id="chatInput"></textarea>
        <button id="sendButton" disabled>Send</button>
        <div id="editStatus" hidden></div>
    `;
}

function textDelta(text) {
    return JSON.stringify({ text_delta: text });
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

    return vi.fn(async url => {
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

    const fetchMock = vi.fn(async url => {
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
            controller.enqueue(encoder.encode(`${textDelta(text)}\n`));
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

// 2) shared test helpers
function setupApp({
    chatPayloads = [[textDelta('Hello')]],
    fetchImpl = null,
    navigatorImpl = {},
    doc = document,
} = {}) {
    const effectiveFetchImpl = fetchImpl || createFetchMock(chatPayloads);
    return createChatApp({
        doc,
        fetchImpl: effectiveFetchImpl,
        navigatorImpl,
    });
}

function setChatInput(app, text) {
    app.elements.chatInput.value = text;
    app.elements.chatInput.dispatchEvent(new Event('input'));
}

function startSendMessage(app, text) {
    setChatInput(app, text);
    return app.sendMessage();
}

async function sendUserMessage(app, text) {
    await startSendMessage(app, text);
}

async function seedConversation(app, userMessages) {
    for (const userMessage of userMessages) {
        await sendUserMessage(app, userMessage);
    }
}

async function flushAsyncWork() {
    await Promise.resolve();
    await Promise.resolve();
}

function asRoleContent(messages) {
    return messages.map(message => ({ role: message.role, content: message.content }));
}

// 3) pure helper tests
describe('parseStreamPayloadLine', () => {
    it('parses text_delta stream payload', () => {
        expect(parseStreamPayloadLine('{"text_delta":"hello"}')).toEqual({ type: 'text_delta', text_delta: 'hello' });
    });

    it('throws on invalid payload json', () => {
        expect(() => parseStreamPayloadLine('{bad json}')).toThrow('Invalid stream payload');
    });
});

describe('buildChatQueryParams', () => {
    it('builds query string with optional geolocation', () => {
        expect(buildChatQueryParams('America/Los_Angeles', null, null)).toBe('time_zone=America%2FLos_Angeles');
        expect(buildChatQueryParams('America/Los_Angeles', 1.23, 4.56)).toContain('latitude=1.23');
        expect(buildChatQueryParams('America/Los_Angeles', 1.23, 4.56)).toContain('longitude=4.56');
    });
});

// 4) createChatApp integration tests
describe('createChatApp messageTree flow', () => {
    beforeEach(() => {
        renderBaseDOM();
    });

    describe('send + stream flow', () => {
        it('sends a message and streams assistant response', async () => {
            const app = setupApp({
                chatPayloads: [[textDelta('Hel'), textDelta('lo')]],
            });

            await sendUserMessage(app, 'Hi');

            expect(asRoleContent(app.getMessages())).toEqual([
                { role: 'user', content: 'Hi' },
                { role: 'assistant', content: 'Hello' },
            ]);
        });
    });

    describe('scroll behavior', () => {
        it('keeps scrolling to the textarea while assistant tokens stream', async () => {
            const app = setupApp({
                chatPayloads: [[textDelta('Hel'), textDelta('lo')]],
            });

            const scrollIntoViewSpy = vi.fn();
            app.elements.chatInput.scrollIntoView = scrollIntoViewSpy;

            await sendUserMessage(app, 'Hi');

            expect(scrollIntoViewSpy).toHaveBeenCalled();
        });

        it('scrolls the chat container to its lowest possible offset', async () => {
            const app = setupApp({
                chatPayloads: [[textDelta('Hello')]],
            });

            const scrollState = mockScrollableContainer(app.elements.chatContainer, {
                clientHeight: 600,
                initialScrollTop: 0,
                scrollHeight: 1600,
            });

            app.elements.chatInput.scrollIntoView = vi.fn();
            await sendUserMessage(app, 'Hi');

            expect(scrollState.getScrollTop()).toBe(scrollState.getMaxScrollTop());
        });

        it('pauses auto-scroll while generating when user scrolls up, then resumes at bottom', async () => {
            const controlledFetch = createControllableFetchMock();
            const app = setupApp({ fetchImpl: controlledFetch.fetchMock });

            const scrollIntoViewSpy = vi.fn();
            app.elements.chatInput.scrollIntoView = scrollIntoViewSpy;
            const scrollState = mockScrollableContainer(app.elements.chatContainer);

            const sendPromise = startSendMessage(app, 'Hi');
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
            const app = setupApp({ fetchImpl: controlledFetch.fetchMock });

            const scrollIntoViewSpy = vi.fn();
            app.elements.chatInput.scrollIntoView = scrollIntoViewSpy;
            const scrollState = mockScrollableContainer(app.elements.chatContainer);

            const sendPromise = startSendMessage(app, 'Hi');
            await controlledFetch.waitForReady();

            controlledFetch.pushTextDelta('first');
            await flushAsyncWork();

            scrollState.setScrollTop(200);
            app.elements.chatContainer.dispatchEvent(new Event('scroll'));
            await flushAsyncWork();

            controlledFetch.pushTextDelta('second');
            await flushAsyncWork();
            const callsWhilePaused = scrollIntoViewSpy.mock.calls.length;

            setChatInput(app, 'draft');
            await flushAsyncWork();
            expect(scrollIntoViewSpy.mock.calls.length).toBeGreaterThan(callsWhilePaused);

            const callsAfterTyping = scrollIntoViewSpy.mock.calls.length;
            controlledFetch.pushTextDelta('third');
            controlledFetch.close();
            await sendPromise;

            expect(scrollIntoViewSpy.mock.calls.length).toBeGreaterThan(callsAfterTyping);
        });

        it('keeps textarea bottom visible while typing multiline input', () => {
            const app = setupApp({
                chatPayloads: [[textDelta('A1')]],
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
            setChatInput(app, 'line1\nline2');

            expect(scrollIntoViewSpy).toHaveBeenCalled();
        });
    });

    describe('edit mode', () => {
        it('scrolls textarea into view when editing a previous user message by click', async () => {
            const app = setupApp({
                chatPayloads: [[textDelta('A1')], [textDelta('A2')]],
            });

            await seedConversation(app, ['U1', 'U2']);

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
            const app = setupApp({
                chatPayloads: [[textDelta('A1')], [textDelta('A2')]],
            });

            await seedConversation(app, ['U1', 'U2']);

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

        it('cancels edit mode on outside mousedown', async () => {
            const app = setupApp({
                chatPayloads: [[textDelta('A1')], [textDelta('A2')]],
            });

            await seedConversation(app, ['U1', 'U2']);

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

    describe('fork navigation', () => {
        it('shows fork buttons and switches between user forks', async () => {
            const app = setupApp({
                chatPayloads: [[textDelta('A1')], [textDelta('A2')], [textDelta('AFork')]],
            });

            await seedConversation(app, ['U1', 'U2']);

            app.editMessage(2);
            await sendUserMessage(app, 'U2 - forked');

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
    });
});
