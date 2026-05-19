/*
 * File organization plan for future changes:
 * 1) constants + exported pure helpers
 * 2) createChatApp setup
 * 3) message-tree/state helpers
 * 4) rendering helpers
 * 5) user actions
 * 6) network/stream side effects
 * 7) event handlers + listener wiring
 * 8) exports + bootstrap
 */

import {
    applyBotMessage,
    applyHandshake,
    applyResyncHandshake,
    buildRequestMessages,
    relabelSnapshotUuid,
} from './conversation_client.js';

// Cap on how many resync round-trips we accept per send. One retry should be
// enough under normal conditions; the second handshake would arrive only if
// new bot history landed between our retry's POST and the handshake reply,
// which is rare. Two retries hard-caps cascading failures.
const MAX_RESYNC_RETRIES = 2;

// 1) constants + exported pure helpers
const MAX_INPUT_HEIGHT = 200;
const BOTTOM_SCROLL_TOLERANCE = 2;
const EDIT_STATUS_TEXT = 'Editing a previous message. Press Esc or click outside to cancel.';
const COPY_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg>';
const CHECK_ICON = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';

export function parseStreamPayloadLine(line) {
    if (!line || !line.trim()) {
        return null;
    }

    let payload;
    try {
        payload = JSON.parse(line);
    } catch {
        throw new Error(`Invalid stream payload: ${line}`);
    }

    if ('text_delta' in payload) {
        if (typeof payload.text_delta !== 'string') {
            throw new Error('Invalid stream payload: text_delta must be a string');
        }
        return { type: 'text_delta', text_delta: payload.text_delta };
    }

    if ('progress_message' in payload) {
        if (typeof payload.progress_message !== 'string') {
            throw new Error('Invalid stream payload: progress_message must be a string');
        }
        return { type: 'progress_message', progress_message: payload.progress_message };
    }

    if ('tool_call' in payload) {
        if (!payload.tool_call || typeof payload.tool_call !== 'object') {
            throw new Error('Invalid stream payload: tool_call must be an object');
        }
        if (typeof payload.tool_call.name !== 'string') {
            throw new Error('Invalid stream payload: tool_call.name must be a string');
        }
        if (
            'arguments' in payload.tool_call
            && typeof payload.tool_call.arguments !== 'string'
        ) {
            throw new Error('Invalid stream payload: tool_call.arguments must be a string');
        }
        return {
            type: 'tool_call',
            tool_call: {
                name: payload.tool_call.name,
                arguments: payload.tool_call.arguments || '{}',
            },
        };
    }

    // The handshake is the first stream event under the unified-conversation wire.
    // It carries `snapshot_uuid` (the authoritative branch for this turn), the
    // `source_id_assignments` map from `client_index` → assigned `source_id` for
    // submitted user nodes, and optionally `unacknowledged_bot_messages` for the
    // post-commit stream-loss recovery path.
    if ('snapshot_uuid' in payload && 'source_id_assignments' in payload) {
        if (typeof payload.snapshot_uuid !== 'string') {
            throw new Error('Invalid stream payload: snapshot_uuid must be a string');
        }
        if (!Array.isArray(payload.source_id_assignments)) {
            throw new Error('Invalid stream payload: source_id_assignments must be an array');
        }
        const handshake = {
            type: 'handshake',
            snapshot_uuid: payload.snapshot_uuid,
            source_id_assignments: payload.source_id_assignments,
        };
        if ('unacknowledged_bot_messages' in payload) {
            if (!Array.isArray(payload.unacknowledged_bot_messages)) {
                throw new Error('Invalid stream payload: unacknowledged_bot_messages must be an array');
            }
            handshake.unacknowledged_bot_messages = payload.unacknowledged_bot_messages;
        }
        return handshake;
    }

    // `bot_message` marks the final commit — the bot's `ConversationMessage` was
    // persisted with the server-assigned `source_id`. The client must wait for
    // this event before creating an assistant node; absence (mid-turn abort, max
    // rounds hit) means no assistant node should be created.
    if ('bot_message' in payload) {
        if (!payload.bot_message || typeof payload.bot_message !== 'object') {
            throw new Error('Invalid stream payload: bot_message must be an object');
        }
        if (typeof payload.bot_message.source_id !== 'string') {
            throw new Error('Invalid stream payload: bot_message.source_id must be a string');
        }
        return { type: 'bot_message', source_id: payload.bot_message.source_id };
    }

    if ('context_pct' in payload) {
        if (typeof payload.context_pct !== 'number') {
            throw new Error('Invalid stream payload: context_pct must be a number');
        }
        return { type: 'context_pct', context_pct: payload.context_pct };
    }

    if ('compaction_pending' in payload) {
        return { type: 'compaction_pending' };
    }

    return { type: 'unknown', payload };
}

export function buildChatQueryParams(timeZone, latitude, longitude) {
    const queryParams = new URLSearchParams({ time_zone: timeZone });
    if (latitude !== null && longitude !== null) {
        queryParams.append('latitude', latitude);
        queryParams.append('longitude', longitude);
    }
    return queryParams.toString();
}

function setInputHeight(inputEl) {
    inputEl.style.height = 'auto';
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, MAX_INPUT_HEIGHT)}px`;
}

function cloneMessages(sourceMessages) {
    return sourceMessages.map(message => ({ ...message }));
}

function humanizeLabel(label) {
    return label
        .replace(/_/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase());
}

// 2) createChatApp setup
export function createChatApp({
    doc = document,
    fetchImpl = fetch,
    navigatorImpl = navigator,
    apiUrl = '',
} = {}) {
    const chatContainer = doc.getElementById('chatContainer');
    const chatWrapper = doc.getElementById('chatWrapper');
    const chatInput = doc.getElementById('chatInput');
    const sendButton = doc.getElementById('sendButton');
    const editStatus = doc.getElementById('editStatus');
    const inputContainer = doc.getElementById('inputContainer');

    if (!chatContainer || !chatWrapper || !chatInput || !sendButton) {
        throw new Error('Chat UI container elements not found');
    }

    if (inputContainer && typeof ResizeObserver !== 'undefined') {
        const ro = new ResizeObserver(() => {
            doc.documentElement.style.setProperty('--input-height', `${inputContainer.offsetHeight}px`);
        });
        ro.observe(inputContainer);
    }

    const md = markdownit({
        html: false,
        linkify: true,
        typographer: true,
        breaks: true,
        highlight(str, lang) {
            const language = lang && hljs.getLanguage(lang) ? lang : null;
            const highlighted = language
                ? hljs.highlight(str, { language, ignoreIllegals: true }).value
                : hljs.highlightAuto(str).value;
            return `<pre><code class="hljs">${highlighted}</code></pre>`;
        },
    });

    function renderMarkdown(content) {
        return DOMPurify.sanitize(md.render(content));
    }

    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;

    let latitude = null;
    let longitude = null;
    if ('geolocation' in navigatorImpl) {
        navigatorImpl.geolocation.getCurrentPosition(
            position => {
                latitude = position.coords.latitude;
                longitude = position.coords.longitude;
            },
            error => {
                console.error(`Failed to get geolocation (${error.code}): ${error.message}`);
            },
            {
                enableHighAccuracy: true,
                maximumAge: 0,
                timeout: 60000,
            },
        );
    }

    let conversationUuid = '';
    let messages = [];
    let isGenerating = false;
    let isAutoScrollSuspended = false;
    let editSession = null;
    let editingParentId = null;
    let editingMessageId = null;
    let nextMessageId = 1;

    const messageTree = new Map();
    const activityByNodeId = new Map();
    messageTree.set(0, {
        id: 0,
        role: 'root',
        content: '',
        parentId: null,
        // Nodes carry `snapshot_uuid` (set by handshake/applyHandshake), and
        // `source_id` (server-assigned). Root has neither.
        snapshot_uuid: null,
        source_id: null,
        children: [],
        activeChildId: null,
    });

    // 3) message-tree/state helpers
    function getNode(nodeId) {
        return messageTree.get(nodeId) || null;
    }

    function createMessageNode(role, content, parentId) {
        const parentNode = getNode(parentId);
        if (!parentNode) {
            return null;
        }

        const node = {
            id: nextMessageId,
            role,
            content,
            parentId,
            // snapshot_uuid + source_id are stamped by applyHandshake / applyBotMessage.
            snapshot_uuid: null,
            source_id: null,
            children: [],
            activeChildId: null,
        };
        nextMessageId += 1;

        messageTree.set(node.id, node);
        parentNode.children.push(node.id);
        parentNode.activeChildId = node.id;
        return node;
    }

    function getPathIdsToNode(nodeId) {
        const pathIds = [];
        let currentId = nodeId;

        while (currentId !== null && currentId !== 0) {
            const currentNode = getNode(currentId);
            if (!currentNode) {
                break;
            }
            pathIds.push(currentId);
            currentId = currentNode.parentId;
        }

        return pathIds.reverse();
    }

    function getActivePathIds() {
        const pathIds = [];
        const rootNode = getNode(0);
        let currentId = rootNode ? rootNode.activeChildId : null;

        while (currentId !== null) {
            const currentNode = getNode(currentId);
            if (!currentNode) {
                break;
            }
            pathIds.push(currentId);
            currentId = currentNode.activeChildId;
        }

        return pathIds;
    }

    function getRenderablePathIds() {
        if (editingMessageId !== null) {
            return getPathIdsToNode(editingMessageId);
        }
        if (editingParentId !== null) {
            return getPathIdsToNode(editingParentId);
        }
        return getActivePathIds();
    }

    function syncRenderableMessages() {
        const pathIds = getRenderablePathIds();
        messages = pathIds.map(pathNodeId => {
            const node = getNode(pathNodeId);
            return {
                id: node.id,
                role: node.role,
                content: node.content,
            };
        });
    }

    function getCurrentLeafId() {
        const activePathIds = getActivePathIds();
        if (activePathIds.length === 0) {
            return 0;
        }
        return activePathIds[activePathIds.length - 1];
    }

    function activateLeaf(leafId) {
        const pathIds = getPathIdsToNode(leafId);
        const rootNode = getNode(0);
        if (!rootNode) {
            return;
        }

        rootNode.activeChildId = pathIds.length > 0 ? pathIds[0] : null;

        for (let i = 0; i < pathIds.length; i += 1) {
            const node = getNode(pathIds[i]);
            if (!node) {
                continue;
            }
            node.activeChildId = i + 1 < pathIds.length ? pathIds[i + 1] : null;
        }
    }

    function getConversationMessagesUpTo(nodeId) {
        const pathIds = getPathIdsToNode(nodeId);
        return pathIds.map(pathNodeId => {
            const node = getNode(pathNodeId);
            return {
                role: node.role,
                content: node.content,
            };
        });
    }

    function getLastSnapshotUuid(upToNodeId) {
        const pathIds = getPathIdsToNode(upToNodeId);
        for (let i = pathIds.length - 1; i >= 0; i -= 1) {
            const node = getNode(pathIds[i]);
            if (node && node.snapshot_uuid) {
                return node.snapshot_uuid;
            }
        }
        return null;
    }

    function getUserSiblingIds(nodeId) {
        const node = getNode(nodeId);
        if (!node || node.role !== 'user' || node.parentId === null) {
            return [];
        }

        const parentNode = getNode(node.parentId);
        if (!parentNode) {
            return [];
        }

        return parentNode.children.filter(childId => {
            const childNode = getNode(childId);
            return Boolean(childNode && childNode.role === 'user');
        });
    }

    function getActivityEntriesForNode(nodeId) {
        return activityByNodeId.get(nodeId) || [];
    }

    function getProgressMessagesForNode(nodeId) {
        return getActivityEntriesForNode(nodeId)
            .filter(entry => entry.type === 'progress')
            .map(entry => entry.text);
    }

    let compactionIndicatorVisible = false;
    let pendingCompactionSnapshotUuid = null;
    let compactionPollInterval = null;

    function startCompactionPolling(snapshotUuid) {
        stopCompactionPolling();
        compactionPollInterval = setInterval(async () => {
            try {
                const params = new URLSearchParams({
                    conversation_uuid: conversationUuid,
                    pending_snapshot_uuid: snapshotUuid,
                });
                const res = await fetchImpl(`${apiUrl}/compaction-status?${params}`);
                if (!res.ok) { stopCompactionPolling(); return; }
                const data = await res.json();
                if (data.done) {
                    if (typeof data.snapshot_uuid === 'string' && data.snapshot_uuid) {
                        relabelSnapshotUuid(messageTree, snapshotUuid, data.snapshot_uuid);
                    }
                    clearCompactionIndicator();
                }
            } catch {
                stopCompactionPolling();
            }
        }, 5_000);
    }

    function stopCompactionPolling() {
        clearInterval(compactionPollInterval);
        compactionPollInterval = null;
    }

    function setContextFill(pct) {
        const el = doc.getElementById('context-fill');
        if (el) {
            const clampedPct = Math.max(0, Math.min(pct, 100));
            el.style.width = `${clampedPct}%`;
            el.title = `${pct}% of context window used`;
        }
    }

    function showCompactionIndicator(snapshotUuid) {
        const el = doc.getElementById('compaction-indicator');
        if (el) {
            el.hidden = false;
        }
        compactionIndicatorVisible = true;
        pendingCompactionSnapshotUuid = snapshotUuid;
        startCompactionPolling(snapshotUuid);
    }

    function clearCompactionIndicator() {
        stopCompactionPolling();
        const el = doc.getElementById('compaction-indicator');
        if (el) {
            el.hidden = true;
        }
        compactionIndicatorVisible = false;
        pendingCompactionSnapshotUuid = null;
    }

    // 4) rendering helpers
    function updateSendButtonState() {
        sendButton.disabled = !chatInput.value.trim() || isGenerating;
    }

    function scrollToBottomWhereInputIs() {
        chatContainer.scrollTop = Math.max(0, chatContainer.scrollHeight - chatContainer.clientHeight);
    }

    function isElementAtBottom(element) {
        if (!element) {
            return true;
        }
        const maxScrollTop = Math.max(0, element.scrollHeight - element.clientHeight);
        return maxScrollTop - element.scrollTop <= BOTTOM_SCROLL_TOLERANCE;
    }

    function isScrolledToBottom() {
        return isElementAtBottom(chatContainer);
    }

    function maybeAutoScrollDuringGeneration() {
        if (!isAutoScrollSuspended) {
            scrollToBottomWhereInputIs();
        }
    }

    function updateEditModeUi() {
        if (editStatus) {
            if (editingMessageId !== null) {
                editStatus.hidden = false;
                editStatus.textContent = EDIT_STATUS_TEXT;
            } else {
                editStatus.hidden = true;
                editStatus.textContent = '';
            }
        }

        if (editingMessageId === null) {
            return;
        }

        const messageIndex = messages.findIndex(message => message.id === editingMessageId);
        if (messageIndex === -1) {
            return;
        }

        const editingMessageEl = chatWrapper.querySelector(
            `.message.user .message-content[data-message-index=\"${messageIndex}\"]`,
        );
        if (editingMessageEl) {
            editingMessageEl.classList.add('editing-source');
        }
    }

    function attachCopyButtons(container) {
        container.querySelectorAll('pre').forEach(pre => {
            if (pre.querySelector('.copy-btn')) {
                return;
            }
            const code = pre.querySelector('code');
            if (!code) {
                return;
            }
            const btn = doc.createElement('button');
            btn.type = 'button';
            btn.className = 'copy-btn';
            btn.setAttribute('aria-label', 'Copy code');
            btn.title = 'Copy code';
            btn.innerHTML = COPY_ICON;
            btn.addEventListener('click', () => {
                navigator.clipboard.writeText(code.textContent).then(() => {
                    btn.innerHTML = CHECK_ICON;
                    setTimeout(() => { btn.innerHTML = COPY_ICON; }, 2000);
                });
            });
            pre.appendChild(btn);
        });
    }

    function appendInlineCode(parent, text) {
        const code = doc.createElement('code');
        code.textContent = String(text);
        parent.appendChild(code);
        return code;
    }

    function appendStrong(parent, text) {
        const strong = doc.createElement('strong');
        strong.textContent = text;
        parent.appendChild(strong);
        return strong;
    }

    function appendTextWithLineBreaks(parent, text) {
        String(text).split('\n').forEach((line, index) => {
            if (index > 0) {
                parent.appendChild(doc.createElement('br'));
            }
            parent.appendChild(doc.createTextNode(line));
        });
    }

    function appendParagraph(container, populate) {
        const paragraph = doc.createElement('p');
        populate(paragraph);
        container.appendChild(paragraph);
        return paragraph;
    }

    function appendTextParagraph(container, text) {
        appendParagraph(container, paragraph => {
            appendTextWithLineBreaks(paragraph, text);
        });
    }

    function appendLabelParagraph(container, label) {
        appendParagraph(container, paragraph => {
            appendStrong(paragraph, label);
        });
    }

    function appendLabelValueParagraph(container, label, value, { inlineCode = false } = {}) {
        appendParagraph(container, paragraph => {
            appendStrong(paragraph, label);
            paragraph.appendChild(doc.createTextNode(': '));
            if (inlineCode) {
                appendInlineCode(paragraph, value);
            } else {
                appendTextWithLineBreaks(paragraph, value);
            }
        });
    }

    function appendCodeBlock(container, text, language = '') {
        const pre = doc.createElement('pre');
        const code = doc.createElement('code');
        code.classList.add('hljs');
        if (language && hljs.getLanguage(language)) {
            code.classList.add(`language-${language}`);
        }
        code.textContent = text;
        pre.appendChild(code);
        container.appendChild(pre);

        try {
            hljs.highlightElement(code);
        } catch {
            code.textContent = text;
        }

        return pre;
    }

    function appendToolCallHeading(container, name) {
        appendParagraph(container, paragraph => {
            paragraph.appendChild(doc.createTextNode('Tool call: '));
            appendInlineCode(paragraph, name);
        });
    }

    function appendGenericToolValue(container, label, value) {
        if (value === null || value === undefined) {
            return false;
        }
        const displayLabel = humanizeLabel(label);
        if (typeof value === 'string') {
            if (!value.trim()) {
                return false;
            }
            if (value.includes('\n')) {
                appendLabelParagraph(container, displayLabel);
                appendCodeBlock(container, value);
                return true;
            }
            appendLabelValueParagraph(container, displayLabel, value);
            return true;
        }
        if (Array.isArray(value)) {
            if (value.length === 0) {
                return false;
            }
            if (value.every(item => ['string', 'number', 'boolean'].includes(typeof item))) {
                appendLabelParagraph(container, displayLabel);
                const list = doc.createElement('ul');
                value.forEach(item => {
                    const listItem = doc.createElement('li');
                    listItem.textContent = String(item);
                    list.appendChild(listItem);
                });
                container.appendChild(list);
                return true;
            }
            appendLabelParagraph(container, displayLabel);
            appendCodeBlock(container, JSON.stringify(value, null, 2), 'json');
            return true;
        }
        if (typeof value === 'object') {
            if (Object.keys(value).length === 0) {
                return false;
            }
            appendLabelParagraph(container, displayLabel);
            appendCodeBlock(container, JSON.stringify(value, null, 2), 'json');
            return true;
        }
        appendLabelValueParagraph(container, displayLabel, String(value), { inlineCode: true });
        return true;
    }

    function appendSortedGenericToolValues(container, parsedArgs, excludedKeys = []) {
        const excluded = new Set(excludedKeys);
        Object.entries(parsedArgs)
            .filter(([key]) => !excluded.has(key))
            .sort(([left], [right]) => left.localeCompare(right))
            .forEach(([key, value]) => {
                appendGenericToolValue(container, key, value);
            });
    }

    function appendGenericToolCallContent(container, name, parsedArgs) {
        appendToolCallHeading(container, name);
        appendSortedGenericToolValues(container, parsedArgs);
    }

    function appendThinkToolCallContent(container, parsedArgs) {
        appendToolCallHeading(container, 'think');
        if (typeof parsedArgs.goal === 'string' && parsedArgs.goal.trim()) {
            appendLabelParagraph(container, 'Goal');
            appendTextParagraph(container, parsedArgs.goal);
        }
        if (typeof parsedArgs.context === 'string' && parsedArgs.context.trim()) {
            appendLabelParagraph(container, 'Context');
            appendTextParagraph(container, parsedArgs.context);
        }
        if (Array.isArray(parsedArgs.perspectives) && parsedArgs.perspectives.length > 0) {
            appendLabelParagraph(container, 'Perspectives');
            const list = doc.createElement('ul');
            parsedArgs.perspectives.forEach(item => {
                const listItem = doc.createElement('li');
                listItem.textContent = String(item);
                list.appendChild(listItem);
            });
            container.appendChild(list);
        }
        appendSortedGenericToolValues(container, parsedArgs, ['goal', 'context', 'perspectives']);
    }

    function appendShellCommandToolCallContent(container, name, parsedArgs) {
        appendToolCallHeading(container, name);
        if (typeof parsedArgs.reason === 'string' && parsedArgs.reason.trim()) {
            appendLabelValueParagraph(container, 'Reason', parsedArgs.reason);
        }
        if (typeof parsedArgs.command === 'string' && parsedArgs.command.trim()) {
            appendLabelParagraph(container, 'Command');
            appendCodeBlock(container, parsedArgs.command, 'sh');
        }
        appendSortedGenericToolValues(container, parsedArgs, ['command', 'reason']);
    }

    function appendFileToolCallContent(container, parsedArgs) {
        const action = parsedArgs.action;
        const path = typeof parsedArgs.path === 'string' ? parsedArgs.path : '';
        const startLine = parsedArgs.start_line;
        const endLine = parsedArgs.end_line;
        const newText = parsedArgs.new_text;
        appendToolCallHeading(container, 'file_tool');
        if (action === 'read_lines') {
            appendParagraph(container, paragraph => {
                paragraph.appendChild(doc.createTextNode('Reading '));
                appendInlineCode(paragraph, path);
                if (typeof startLine === 'number') {
                    paragraph.appendChild(doc.createTextNode(` from line ${startLine}`));
                }
            });
        } else if (action === 'create_file') {
            appendParagraph(container, paragraph => {
                paragraph.appendChild(doc.createTextNode('Creating '));
                appendInlineCode(paragraph, path);
            });
            if (typeof newText === 'string' && newText.length > 0) {
                appendCodeBlock(container, newText);
            }
        } else if (action === 'replace_lines') {
            appendParagraph(container, paragraph => {
                paragraph.appendChild(doc.createTextNode('Editing '));
                appendInlineCode(paragraph, path);
                paragraph.appendChild(doc.createTextNode(` lines ${startLine}-${endLine}`));
            });
            if (typeof newText === 'string' && newText.length > 0) {
                appendCodeBlock(container, newText);
            }
        } else if (action === 'insert_lines') {
            appendParagraph(container, paragraph => {
                paragraph.appendChild(doc.createTextNode('Inserting at '));
                appendInlineCode(paragraph, path);
                paragraph.appendChild(doc.createTextNode(` line ${startLine}`));
            });
            if (typeof newText === 'string' && newText.length > 0) {
                appendCodeBlock(container, newText);
            }
        } else if (action === 'delete_lines') {
            appendParagraph(container, paragraph => {
                paragraph.appendChild(doc.createTextNode('Deleting '));
                appendInlineCode(paragraph, path);
                paragraph.appendChild(doc.createTextNode(` lines ${startLine}-${endLine}`));
            });
        } else {
            const actionLabel = typeof action === 'string' && action ? action : 'unknown';
            appendParagraph(container, paragraph => {
                paragraph.appendChild(doc.createTextNode('Unknown file action: '));
                appendInlineCode(paragraph, actionLabel);
            });
        }
    }

    function appendToolArgumentsFallback(container, name, argumentsText) {
        appendToolCallHeading(container, name);
        appendLabelParagraph(container, 'Arguments');
        appendCodeBlock(container, argumentsText, 'json');
    }

    function appendToolCallContent(container, name, rawArguments) {
        let parsedArgs;
        try {
            parsedArgs = JSON.parse(rawArguments || '{}');
        } catch {
            appendToolArgumentsFallback(container, name, rawArguments || '{}');
            return;
        }

        if (!parsedArgs || typeof parsedArgs !== 'object' || Array.isArray(parsedArgs)) {
            appendToolArgumentsFallback(container, name, JSON.stringify(parsedArgs, null, 2));
            return;
        }

        if (name === 'file_tool') {
            appendFileToolCallContent(container, parsedArgs);
            return;
        }
        if (name === 'shell_command' || name === 'shell_tool') {
            appendShellCommandToolCallContent(container, name, parsedArgs);
            return;
        }
        if (name === 'think') {
            appendThinkToolCallContent(container, parsedArgs);
            return;
        }
        appendGenericToolCallContent(container, name, parsedArgs);
    }

    function renderActivityEntry(entry) {
        const activityDiv = doc.createElement('div');
        activityDiv.className = `message-activity message-activity-${entry.type}`;
        if (entry.type === 'progress') {
            activityDiv.innerHTML = renderMarkdown(entry.text);
        } else if (entry.type === 'tool_call') {
            appendToolCallContent(activityDiv, entry.name, entry.arguments);
            attachCopyButtons(activityDiv);
        }
        return activityDiv;
    }

    function addMessage(role, content, messageIndex = null) {
        const messageDiv = doc.createElement('div');
        messageDiv.className = `message ${role}`;

        const messageData = messageIndex !== null ? messages[messageIndex] : null;
        const messageNode = messageData ? getNode(messageData.id) : null;
        if (role === 'assistant' && messageNode) {
            const activityEntries = getActivityEntriesForNode(messageNode.id);
            if (activityEntries.length > 0) {
                activityEntries.forEach(entry => {
                    messageDiv.appendChild(renderActivityEntry(entry));
                });
            }
        }

        const contentDiv = doc.createElement('div');
        contentDiv.className = 'message-content';
        if (role === 'user' || role === 'assistant') {
            contentDiv.innerHTML = renderMarkdown(content);
        } else {
            contentDiv.textContent = content;
        }
        if (role === 'assistant') {
            attachCopyButtons(contentDiv);
        }

        if (role === 'user' && messageNode) {
            contentDiv.setAttribute('data-message-index', messageIndex);
            contentDiv.setAttribute('title', 'Click to edit and restart from here');
            contentDiv.addEventListener('click', () => {
                if (!isGenerating) {
                    editMessage(messageIndex);
                }
            });
        }

        messageDiv.appendChild(contentDiv);

        let regenBtn = null;
        if (role === 'assistant' && messageNode) {
            const actionsDiv = doc.createElement('div');
            actionsDiv.className = 'message-actions';

            regenBtn = doc.createElement('button');
            regenBtn.type = 'button';
            regenBtn.className = 'icon-btn';
            regenBtn.setAttribute('aria-label', 'Regenerate response');
            regenBtn.title = 'Regenerate this response';
            regenBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 12a9 9 0 0 1-9 9 9 9 0 0 1-6.36-2.64"/>
                    <path d="M3 12a9 9 0 0 1 9-9 9 9 0 0 1 6.36 2.64"/>
                    <path d="M21 3v6h-6"/>
                    <path d="M3 21v-6h6"/>
                </svg>
            `;

            regenBtn.disabled = isGenerating;
            regenBtn.addEventListener('click', event => {
                event.stopPropagation();
                if (!isGenerating) {
                    void regenerateMessage(messageIndex);
                }
            });

            actionsDiv.appendChild(regenBtn);
            messageDiv.appendChild(actionsDiv);
        }

        if (role === 'user' && messageNode) {
            const siblingUserIds = getUserSiblingIds(messageNode.id);
            if (siblingUserIds.length > 1) {
                const currentIndex = siblingUserIds.indexOf(messageNode.id);
                if (currentIndex !== -1) {
                    const actionsDiv = doc.createElement('div');
                    actionsDiv.className = 'message-actions';

                    const prevBtn = doc.createElement('button');
                    prevBtn.type = 'button';
                    prevBtn.className = 'icon-btn fork-nav-btn';
                    prevBtn.setAttribute('aria-label', 'Previous fork');
                    prevBtn.title = 'Previous fork';
                    prevBtn.textContent = '⟨';
                    prevBtn.disabled = isGenerating || currentIndex === 0;
                    prevBtn.addEventListener('click', event => {
                        event.stopPropagation();
                        switchUserFork(messageIndex, -1);
                    });

                    const forkIndicator = doc.createElement('span');
                    forkIndicator.className = 'fork-indicator';
                    forkIndicator.textContent = `${currentIndex + 1}/${siblingUserIds.length}`;

                    const nextBtn = doc.createElement('button');
                    nextBtn.type = 'button';
                    nextBtn.className = 'icon-btn fork-nav-btn';
                    nextBtn.setAttribute('aria-label', 'Next fork');
                    nextBtn.title = 'Next fork';
                    nextBtn.textContent = '⟩';
                    nextBtn.disabled = isGenerating || currentIndex === siblingUserIds.length - 1;
                    nextBtn.addEventListener('click', event => {
                        event.stopPropagation();
                        switchUserFork(messageIndex, 1);
                    });

                    actionsDiv.appendChild(prevBtn);
                    actionsDiv.appendChild(forkIndicator);
                    actionsDiv.appendChild(nextBtn);
                    messageDiv.appendChild(actionsDiv);
                }
            }
        }

        chatWrapper.appendChild(messageDiv);

        return { contentDiv, regenBtn };
    }

    function renderMessages({ scrollToBottom = true } = {}) {
        const previousChatScrollTop = chatContainer.scrollTop;

        syncRenderableMessages();
        chatWrapper.innerHTML = '';
        messages.forEach((message, index) => {
            addMessage(message.role, message.content, index);
        });
        updateEditModeUi();

        if (scrollToBottom) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return;
        }

        chatContainer.scrollTop = previousChatScrollTop;
    }

    // 5) user actions
    function cancelEditSession() {
        if (!editSession || isGenerating) {
            return false;
        }

        activateLeaf(editSession.activeLeafId);
        editingParentId = editSession.editingParentId;
        editingMessageId = editSession.editingMessageId;
        chatInput.value = editSession.inputValue;
        setInputHeight(chatInput);

        editSession = null;
        renderMessages();
        updateSendButtonState();
        return true;
    }

    function switchUserFork(messageIndex, direction) {
        if (messageIndex < 0 || messageIndex >= messages.length || isGenerating) {
            return;
        }

        const messageData = messages[messageIndex];
        const currentNode = getNode(messageData.id);
        if (!currentNode || currentNode.role !== 'user' || currentNode.parentId === null) {
            return;
        }

        const parentNode = getNode(currentNode.parentId);
        if (!parentNode) {
            return;
        }

        const siblingUserIds = getUserSiblingIds(currentNode.id);
        const currentIndex = siblingUserIds.indexOf(currentNode.id);
        if (currentIndex === -1) {
            return;
        }

        const targetIndex = currentIndex + direction;
        if (targetIndex < 0 || targetIndex >= siblingUserIds.length) {
            return;
        }

        parentNode.activeChildId = siblingUserIds[targetIndex];
        editingParentId = null;
        editingMessageId = null;
        editSession = null;
        renderMessages();
        updateSendButtonState();
    }

    function editMessage(messageIndex) {
        if (messageIndex < 0 || messageIndex >= messages.length) {
            return;
        }

        const messageToEdit = messages[messageIndex];
        if (messageToEdit.role !== 'user') {
            return;
        }

        const messageNode = getNode(messageToEdit.id);
        if (!messageNode || messageNode.parentId === null) {
            return;
        }

        if (!editSession) {
            editSession = {
                activeLeafId: getCurrentLeafId(),
                inputValue: chatInput.value,
                editingParentId,
                editingMessageId,
            };
        }

        chatInput.value = messageToEdit.content;
        setInputHeight(chatInput);
        editingParentId = messageNode.parentId;
        editingMessageId = messageNode.id;
        renderMessages();

        sendButton.disabled = false;
        chatInput.focus();
        scrollToBottomWhereInputIs();
    }

    async function regenerateMessage(messageIndex) {
        if (messageIndex < 0 || messageIndex >= messages.length) {
            return;
        }

        const messageToRegenerate = messages[messageIndex];
        if (messageToRegenerate.role !== 'assistant') {
            return;
        }

        const assistantNode = getNode(messageToRegenerate.id);
        if (!assistantNode || assistantNode.parentId === null) {
            return;
        }

        const parentNode = getNode(assistantNode.parentId);
        if (!parentNode) {
            return;
        }

        parentNode.activeChildId = null;
        editingParentId = null;
        editingMessageId = null;
        editSession = null;
        renderMessages();

        // Regenerate has no pending user node — the user message already exists
        // and was acked. On resync, applyResyncHandshake reconstructs unacked
        // bots but pops nothing.
        await generateAssistantResponse(parentNode.id, {
            composeMode: 'regenerate',
            pendingUserNodeId: null,
        });
    }

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message || isGenerating) {
            return;
        }

        // "edit" mode = user typed a new fork under a non-leaf node.
        // "send-from-leaf" = user typed at the tail of the active path.
        const composeMode = editingParentId !== null ? 'edit' : 'send-from-leaf';
        const parentNodeId = editingParentId !== null ? editingParentId : getCurrentLeafId();

        editSession = null;
        editingParentId = null;
        editingMessageId = null;
        chatInput.value = '';
        chatInput.style.height = 'auto';

        const userNode = createMessageNode('user', message, parentNodeId);
        if (!userNode) {
            updateSendButtonState();
            return;
        }

        renderMessages();
        await generateAssistantResponse(userNode.id, {
            composeMode,
            pendingUserNodeId: userNode.id,
        });
    }

    // 6) network/stream side effects
    async function generateAssistantResponse(parentNodeId, {
        composeMode = 'send-from-leaf',
        pendingUserNodeId = null,
        resyncRetryCount = 0,
    } = {}) {
        isGenerating = true;
        isAutoScrollSuspended = false;
        sendButton.disabled = true;

        let didCompleteResponse = false;
        // Captures resync state observed inside the stream so the finally / post-loop
        // block can dispatch the right recovery action without re-parsing.
        let resyncState = null;

        const pendingMessageDiv = doc.createElement('div');
        pendingMessageDiv.className = 'message assistant';

        const activityContainer = doc.createElement('div');
        activityContainer.hidden = true;
        pendingMessageDiv.appendChild(activityContainer);

        const assistantContent = doc.createElement('div');
        assistantContent.className = 'message-content';
        pendingMessageDiv.appendChild(assistantContent);

        const loadingIndicator = doc.createElement('div');
        loadingIndicator.className = 'loading-indicator';
        loadingIndicator.innerHTML = '<span class="typing-indicator"></span>';
        pendingMessageDiv.appendChild(loadingIndicator);
        chatWrapper.appendChild(pendingMessageDiv);

        maybeAutoScrollDuringGeneration();

        // Capture the client-side node ids in the same order they appear in the
        // request payload. The handshake's `source_id_assignments` references
        // positions in this array via `client_index`.
        const sentClientIds = getPathIdsToNode(parentNodeId);
        try {
            const query = buildChatQueryParams(timeZone, latitude, longitude);
            const response = await fetchImpl(`${apiUrl}/chat?${query}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    conversation_uuid: conversationUuid,
                    snapshot_uuid: getLastSnapshotUuid(parentNodeId),
                    // `buildRequestMessages` emits `source_id` only for server-stamped
                    // nodes, so the DAG-scoped guardrail recognizes the echoed assistant
                    // entries on multi-turn POSTs.
                    messages: buildRequestMessages(messageTree, sentClientIds),
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                const errorDetail = error?.detail || response.statusText;
                throw new Error(`HTTP ${response.status} error: ${errorDetail}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';
            const activityEntries = [];
            let streamBuffer = '';
            let receivedSnapshotUuid = null;
            // `bot_message` source_id captured from the stream. Required before
            // the assistant node is created — its presence is the commit signal.
            let receivedBotSourceId = null;

            const renderPendingActivity = () => {
                activityContainer.innerHTML = '';
                activityEntries.forEach(entry => {
                    activityContainer.appendChild(renderActivityEntry(entry));
                });
                activityContainer.hidden = activityEntries.length === 0;
            };

            const processLine = line => {
                const parsed = parseStreamPayloadLine(line);
                if (!parsed) {
                    return;
                }

                if (parsed.type === 'handshake') {
                    // Stamp snapshot_uuid + assigned source_ids onto submitted user
                    // nodes. Note: no legacy "different uuid clears indicator"
                    // behavior — the polling endpoint is the only clear path now,
                    // so a sibling-branch send during pending compaction must NOT
                    // disturb the indicator.
                    receivedSnapshotUuid = parsed.snapshot_uuid;
                    applyHandshake(messageTree, sentClientIds, parsed);
                    console.log('Snapshot UUID:', receivedSnapshotUuid);
                    // Resync handshake: the server detected post-commit stream loss
                    // and is closing the stream without invoking the LLM. We need
                    // to apply the resync repair (reconstruct unacked bots, repair
                    // the pending user per composeMode) once the stream is fully
                    // drained, then either auto-retry or restore the draft.
                    if (
                        Array.isArray(parsed.unacknowledged_bot_messages)
                        && parsed.unacknowledged_bot_messages.length > 0
                    ) {
                        const result = applyResyncHandshake(messageTree, {
                            pendingUserNodeId,
                            handshake: parsed,
                            composeMode,
                            createNodeFn: (role, content, parentId) => {
                                const created = createMessageNode(role, content, parentId);
                                return created ? created.id : null;
                            },
                        });
                        resyncState = {
                            assistantNodeIds: result.assistantNodeIds,
                            draftContent: result.draftContent,
                        };
                    }
                    return;
                }

                if (parsed.type === 'bot_message') {
                    // Final-commit marker. Capture the assigned source_id; the
                    // assistant node is created post-loop only when this is set.
                    receivedBotSourceId = parsed.source_id;
                    return;
                }

                if (parsed.type === 'context_pct') {
                    setContextFill(parsed.context_pct);
                    return;
                }

                if (parsed.type === 'compaction_pending') {
                    showCompactionIndicator(receivedSnapshotUuid);
                    return;
                }

                if (parsed.type === 'progress_message') {
                    activityEntries.push({ type: 'progress', text: parsed.progress_message });
                    renderPendingActivity();
                    maybeAutoScrollDuringGeneration();
                    return;
                }

                if (parsed.type === 'tool_call') {
                    activityEntries.push({
                        type: 'tool_call',
                        name: parsed.tool_call.name,
                        arguments: parsed.tool_call.arguments,
                    });
                    renderPendingActivity();
                    maybeAutoScrollDuringGeneration();
                    return;
                }

                if (parsed.type === 'text_delta') {
                    fullResponse += parsed.text_delta;
                    assistantContent.innerHTML = renderMarkdown(fullResponse);
                    maybeAutoScrollDuringGeneration();
                }
            };

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                streamBuffer += decoder.decode(value, { stream: true });
                let newlineIndex = streamBuffer.indexOf('\n');
                while (newlineIndex !== -1) {
                    const line = streamBuffer.slice(0, newlineIndex).trim();
                    streamBuffer = streamBuffer.slice(newlineIndex + 1);
                    processLine(line);
                    newlineIndex = streamBuffer.indexOf('\n');
                }
            }

            streamBuffer += decoder.decode();
            const finalLine = streamBuffer.trim();
            if (finalLine) {
                processLine(finalLine);
            }

            // The assistant node is created only when the server emitted
            // `bot_message` — that's the exactly-once commit signal. If the stream
            // ended without one (mid-turn abort, max-rounds hit, error), no
            // assistant node should appear in the tree.
            let assistantNode = null;
            if (receivedBotSourceId) {
                const newNodeId = applyBotMessage(messageTree, {
                    parentNodeId,
                    fullResponse,
                    snapshotUuid: receivedSnapshotUuid,
                    sourceId: receivedBotSourceId,
                    // applyBotMessage delegates node creation back to createMessageNode
                    // so children/activeChildId are wired correctly by the existing
                    // tree code (it requires the parent's children/activeChildId update).
                    createNodeFn: (role, content, parentId) => {
                        const created = createMessageNode(role, content, parentId);
                        return created ? created.id : null;
                    },
                });
                assistantNode = getNode(newNodeId);
            }
            if (assistantNode && activityEntries.length > 0) {
                activityByNodeId.set(assistantNode.id, [...activityEntries]);
            }
            didCompleteResponse = Boolean(assistantNode);
        } catch (error) {
            if (compactionIndicatorVisible) {
                clearCompactionIndicator();
            }
            console.error('Error:', error);
            activityContainer.hidden = true;
            assistantContent.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
        } finally {
            loadingIndicator.style.display = 'none';
            isGenerating = false;
            updateSendButtonState();
            if (didCompleteResponse) {
                renderMessages({ scrollToBottom: !isAutoScrollSuspended });
                if (!isAutoScrollSuspended) {
                    scrollToBottomWhereInputIs();
                }
            } else if (resyncState) {
                // Resync path: no commit signal arrived, so the pending DOM bubble
                // must be removed explicitly. renderMessages() rebuilds the tree
                // view from the active path, so the reconstructed unacked bots
                // (and reparented pending user, for send-from-leaf) appear.
                if (pendingMessageDiv.parentNode) {
                    pendingMessageDiv.parentNode.removeChild(pendingMessageDiv);
                }
                renderMessages({ scrollToBottom: !isAutoScrollSuspended });
                if (composeMode === 'send-from-leaf' && resyncRetryCount < MAX_RESYNC_RETRIES) {
                    // Auto-retry: the pending user node is now reparented under
                    // the recovered bot; its source_id will get assigned on the
                    // retry's handshake. Re-use the same pendingUserNodeId since
                    // it's still the parent for the bot's reply.
                    await generateAssistantResponse(pendingUserNodeId, {
                        composeMode,
                        pendingUserNodeId,
                        resyncRetryCount: resyncRetryCount + 1,
                    });
                } else if (
                    (composeMode === 'edit' || composeMode === 'regenerate')
                    && typeof resyncState.draftContent === 'string'
                    && resyncState.draftContent
                ) {
                    // Restore the popped pending user's content to the input box
                    // so the user can re-author against the recovered state.
                    chatInput.value = resyncState.draftContent;
                    setInputHeight(chatInput);
                    updateSendButtonState();
                }
            }
        }
    }

    // 7) event handlers + listener wiring
    function handleScrollDuringGeneration() {
        if (!isGenerating) {
            return;
        }
        isAutoScrollSuspended = !isScrolledToBottom();
    }

    function handleInput() {
        setInputHeight(chatInput);
        if (isGenerating && isAutoScrollSuspended) {
            isAutoScrollSuspended = false;
            scrollToBottomWhereInputIs();
        }
        updateSendButtonState();
    }

    function handleKeyDown(event) {
        if (event.key === 'Escape') {
            if (cancelEditSession()) {
                event.preventDefault();
            }
            return;
        }

        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            void sendMessage();
        }
    }

    function handleDocumentMouseDown(event) {
        if (!editSession || isGenerating) {
            return;
        }

        const target = event.target;
        const clickInsideInput = target === chatInput || chatInput.contains(target);
        const clickOnSendButton = target === sendButton || sendButton.contains(target);
        if (!clickInsideInput && !clickOnSendButton) {
            cancelEditSession();
        }
    }

    chatInput.addEventListener('input', handleInput);

    chatContainer.addEventListener('scroll', handleScrollDuringGeneration, { passive: true });

    chatInput.addEventListener('keydown', handleKeyDown);
    sendButton.addEventListener('click', () => {
        void sendMessage();
    });
    doc.addEventListener('mousedown', handleDocumentMouseDown);

    updateSendButtonState();
    updateEditModeUi();
    chatInput.focus();

    fetchImpl(`${apiUrl}/conversation`)
        .then(response => response.json())
        .then(data => {
            conversationUuid = data.conversation_uuid;
            console.log('Conversation UUID:', conversationUuid);
        })
        .catch(error => {
            console.error('Service not available:', error);
            chatWrapper.innerHTML = '<div class="error-message">Service not available.</div>';
        });

    return {
        cancelEditSession,
        editMessage,
        generateAssistantResponse,
        getIsEditing: () => editSession !== null,
        getActivityEntriesForNode,
        getMessages: () => cloneMessages(messages),
        getNode,
        getProgressMessagesForNode,
        getUserSiblingIds,
        regenerateMessage,
        sendMessage,
        switchUserFork,
        handleKeyDown,
        elements: {
            chatContainer,
            chatWrapper,
            chatInput,
            sendButton,
            editStatus,
        },
    };
}

// 8) exports + bootstrap
export function initChatUI(options) {
    return createChatApp(options);
}

if (
    typeof document !== 'undefined' &&
    document.getElementById('chatContainer') &&
    document.getElementById('chatWrapper') &&
    document.getElementById('chatInput') &&
    document.getElementById('sendButton')
) {
    initChatUI();
}
