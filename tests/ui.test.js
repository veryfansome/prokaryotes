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
