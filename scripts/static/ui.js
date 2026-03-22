const MAX_INPUT_HEIGHT = 200;
const BOTTOM_SCROLL_TOLERANCE = 2;
const EDIT_STATUS_TEXT = 'Editing a previous message. Press Esc or click outside to cancel.';

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

    if ('prompt_uuid' in payload) {
        return { type: 'prompt', payload };
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

    if (!chatContainer || !chatWrapper || !chatInput || !sendButton) {
        throw new Error('Chat UI container elements not found');
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
    messageTree.set(0, {
        id: 0,
        role: 'root',
        content: '',
        parentId: null,
        children: [],
        activeChildId: null,
    });

    function updateSendButtonState() {
        sendButton.disabled = !chatInput.value.trim() || isGenerating;
    }

    function scrollToBottomWhereInputIs() {
        const scrollToLowestOffsets = () => {
            chatContainer.scrollTop = Math.max(0, chatContainer.scrollHeight - chatContainer.clientHeight);
            const scrollingElement = doc.scrollingElement;
            if (scrollingElement) {
                scrollingElement.scrollTop = Math.max(0, scrollingElement.scrollHeight - scrollingElement.clientHeight);
            }
        };

        if (typeof chatInput.scrollIntoView === 'function') {
            try {
                chatInput.scrollIntoView({ block: 'end', inline: 'nearest' });
            } catch {
                // Ignore scroll API failures in non-browser environments (e.g. tests).
            }
        }

        scrollToLowestOffsets();
        // Run twice to account for layout settling during streaming reflows.
        void chatContainer.offsetHeight;
        scrollToLowestOffsets();
    }

    function isElementAtBottom(element) {
        if (!element) {
            return true;
        }
        const maxScrollTop = Math.max(0, element.scrollHeight - element.clientHeight);
        return maxScrollTop - element.scrollTop <= BOTTOM_SCROLL_TOLERANCE;
    }

    function isScrolledToBottom() {
        return isElementAtBottom(chatContainer) && isElementAtBottom(doc.scrollingElement);
    }

    function maybeAutoScrollDuringGeneration() {
        if (!isAutoScrollSuspended) {
            scrollToBottomWhereInputIs();
        }
    }

    function handleScrollDuringGeneration() {
        if (!isGenerating) {
            return;
        }
        isAutoScrollSuspended = !isScrolledToBottom();
    }

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

    function addMessage(role, content, messageIndex = null) {
        const messageDiv = doc.createElement('div');
        messageDiv.className = `message ${role}`;

        const contentDiv = doc.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        const messageData = messageIndex !== null ? messages[messageIndex] : null;
        const messageNode = messageData ? getNode(messageData.id) : null;

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
        const scrollingElement = doc.scrollingElement;
        const previousPageScrollTop = scrollingElement ? scrollingElement.scrollTop : 0;

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
        if (scrollingElement) {
            scrollingElement.scrollTop = previousPageScrollTop;
        }
    }

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

    async function generateAssistantResponse(parentNodeId) {
        isGenerating = true;
        isAutoScrollSuspended = false;
        sendButton.disabled = true;

        let didCompleteResponse = false;

        const pendingMessageDiv = doc.createElement('div');
        pendingMessageDiv.className = 'message assistant';
        const assistantContent = doc.createElement('div');
        assistantContent.className = 'message-content';
        assistantContent.innerHTML = '<span class="typing-indicator"></span>';
        pendingMessageDiv.appendChild(assistantContent);
        chatWrapper.appendChild(pendingMessageDiv);
        maybeAutoScrollDuringGeneration();

        try {
            const query = buildChatQueryParams(timeZone, latitude, longitude);
            const response = await fetchImpl(`${apiUrl}/chat?${query}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    conversation_uuid: conversationUuid,
                    messages: getConversationMessagesUpTo(parentNodeId),
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
            let streamBuffer = '';

            const processLine = line => {
                const parsed = parseStreamPayloadLine(line);
                if (!parsed) {
                    return;
                }

                if (parsed.type === 'text_delta') {
                    if (fullResponse === '') {
                        assistantContent.textContent = '';
                    }
                    fullResponse += parsed.text_delta;
                    assistantContent.textContent = fullResponse;
                    maybeAutoScrollDuringGeneration();
                    return;
                }

                if (parsed.type === 'prompt') {
                    console.log('Prompt:', parsed.payload);
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

            createMessageNode('assistant', fullResponse, parentNodeId);
            didCompleteResponse = true;
        } catch (error) {
            console.error('Error:', error);
            assistantContent.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
        } finally {
            isGenerating = false;
            updateSendButtonState();
            if (didCompleteResponse) {
                renderMessages({ scrollToBottom: !isAutoScrollSuspended });
                if (!isAutoScrollSuspended) {
                    scrollToBottomWhereInputIs();
                }
            }
        }
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

        await generateAssistantResponse(parentNode.id);
    }

    async function sendMessage() {
        const message = chatInput.value.trim();
        if (!message || isGenerating) {
            return;
        }

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
        await generateAssistantResponse(userNode.id);
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

    chatInput.addEventListener('input', () => {
        if (isGenerating && isAutoScrollSuspended) {
            isAutoScrollSuspended = false;
            scrollToBottomWhereInputIs();
        }
        setInputHeight(chatInput);
        updateSendButtonState();
    });

    chatContainer.addEventListener('scroll', handleScrollDuringGeneration, { passive: true });
    const scrollingElement = doc.scrollingElement;
    if (scrollingElement && scrollingElement !== chatContainer) {
        scrollingElement.addEventListener('scroll', handleScrollDuringGeneration, { passive: true });
    }
    if (doc.defaultView) {
        doc.defaultView.addEventListener('scroll', handleScrollDuringGeneration, { passive: true });
    }

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
            console.log('Conversation:', data);
            conversationUuid = data.conversation_uuid;
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
        getMessages: () => cloneMessages(messages),
        getNode,
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
