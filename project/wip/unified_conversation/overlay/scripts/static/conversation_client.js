// Client-side primitives for the unified conversation wire protocol.
//
// These functions are pure (no DOM, no fetch, no global state) so they can be
// unit-tested with Vitest. The existing ui.js calls into them at three points:
//
// 1. **Handshake** (first stream event): `applyHandshake(messageTree, sentClientIds, handshake)`
//    stamps server-assigned `source_id`s onto pending user nodes and sets
//    `snapshot_uuid` on every node submitted in this request (per the invariant
//    "every node with a source_id also has a snapshot_uuid, set at the same time").
//
// 2. **Bot message** (last persistence-relevant event): `applyBotMessage(...)`
//    creates the assistant node and stamps both ids on it.
//
// 3. **Compaction relabel** (driven by `/compaction-status` polling):
//    `relabelSnapshotUuid(messageTree, old, new)` walks the tree idempotently
//    and updates only nodes carrying `snapshot_uuid === old`.
//
// The resync handshake (post-commit stream loss recovery) is handled by
// `applyResyncHandshake(messageTree, pendingUserNodeId, handshake, composeMode)`.
//
// `messageTree` here is a `Map<nodeId, node>` to match `scripts/static/ui.js`'s
// production data structure (`const messageTree = new Map()`).

/**
 * Stamp `source_id` + `snapshot_uuid` on user nodes submitted in this POST.
 *
 * `sentClientIds` is the list of client-side node ids in the same order they
 * appeared in the request payload. The handshake's `source_id_assignments`
 * references positions in that array via `client_index`. Every node in
 * `sentClientIds` (whether or not it received an assignment) gets the
 * handshake's `snapshot_uuid` — that's the authoritative branch for this turn.
 *
 * Invariant: every node that has a `source_id` also has a `snapshot_uuid` set
 * at the same moment. We enforce this by stamping both fields together.
 *
 * Returns the number of nodes stamped.
 */
export function applyHandshake(messageTree, sentClientIds, handshake) {
    if (!handshake || !handshake.snapshot_uuid) {
        return 0;
    }
    const assignments = handshake.source_id_assignments || [];
    const assignmentByIndex = new Map();
    for (const a of assignments) {
        assignmentByIndex.set(a.client_index, a.source_id);
    }
    let stamped = 0;
    for (let i = 0; i < sentClientIds.length; i++) {
        const nodeId = sentClientIds[i];
        const node = messageTree.get(nodeId);
        if (!node) {
            continue;
        }
        // Stamp the snapshot_uuid on every submitted node — even ones that already
        // had a source_id from a prior turn — because a Divergence may have created
        // a fresh branch the existing source_ids now belong to.
        node.snapshot_uuid = handshake.snapshot_uuid;
        const assigned = assignmentByIndex.get(i);
        if (assigned && !node.source_id) {
            node.source_id = assigned;
        }
        stamped += 1;
    }
    return stamped;
}

/**
 * Create the assistant node from the accumulated stream text, stamp both ids.
 *
 * Called on the `bot_message` event (the exactly-once final persistence-
 * relevant event). The handshake's `snapshot_uuid` is the authoritative branch
 * the bot replied into; the `bot_message.source_id` identifies the bot's
 * ConversationMessage.
 *
 * Returns the new node id.
 */
export function applyBotMessage(messageTree, {
    parentNodeId,
    fullResponse,
    snapshotUuid,
    sourceId,
    createNodeFn,
}) {
    const newNodeId = createNodeFn("assistant", fullResponse, parentNodeId);
    const node = messageTree.get(newNodeId);
    if (node) {
        node.source_id = sourceId;
        node.snapshot_uuid = snapshotUuid;
    }
    return newNodeId;
}

/**
 * Idempotent walk: update only nodes whose `snapshot_uuid === oldUuid`.
 *
 * Used by the compaction-status poller to re-label pre-compaction tree nodes
 * onto the child snapshot. Sibling branches under the same `conversation_uuid`
 * keep their own labels and are unaffected. Safe under polling races, repeated
 * `{done: true}` responses, and back-to-back compactions.
 *
 * Returns the number of nodes updated.
 */
export function relabelSnapshotUuid(messageTree, oldUuid, newUuid) {
    if (!oldUuid || !newUuid || oldUuid === newUuid) {
        return 0;
    }
    let count = 0;
    for (const node of messageTree.values()) {
        if (node && node.snapshot_uuid === oldUuid) {
            node.snapshot_uuid = newUuid;
            count += 1;
        }
    }
    return count;
}

/**
 * Apply a resync handshake: reconstruct unacknowledged assistant nodes under
 * their `parent_source_id`, then dispatch the pending user node per
 * `composeMode`.
 *
 * - `composeMode === "send-from-leaf"`: reparent the pending user node so its
 *   parent becomes the last (most recent in `source_id` order) reconstructed
 *   assistant node. Maintains bidirectional links — removes the pending node
 *   from its old parent's `children` array, appends it to the new parent's
 *   `children`, and sets the new parent's `activeChildId` to keep the
 *   reparented node as the active leaf. Caller is expected to **auto-retry**
 *   the original POST after this returns.
 *
 * - `composeMode === "edit"` or `"regenerate"`: pop the pending user node out
 *   of the tree (caller restores its content to the draft input) AND remove
 *   its id from its parent's `children` array so fork navigation doesn't see
 *   a dangling reference. Do not auto-retry; the user authored the message
 *   without seeing the recovered bot history, so silently auto-sending would
 *   change their intent.
 *
 * Returns `{ assistantNodeIds, draftContent }`. On send-from-leaf,
 * `draftContent` is null. On edit/regenerate, `draftContent` is the popped
 * pending user node's content so the caller can restore it to the input box.
 */
export function applyResyncHandshake(messageTree, {
    pendingUserNodeId,
    handshake,
    composeMode,
    createNodeFn,
}) {
    const snapshotUuid = handshake.snapshot_uuid;
    const unacked = handshake.unacknowledged_bot_messages || [];
    // Reconstruct assistant nodes in source_id ascending order. Chained entries
    // (a later entry whose parent_source_id matches an earlier entry's source_id)
    // become a chain because each is created under its parent_source_id's node.
    // `createNodeFn` is expected to wire `children` / `activeChildId` on the
    // parent (production `createMessageNode` does this); we don't repair the
    // newly-created chain here.
    const sourceIdToNodeId = _indexBySourceId(messageTree);
    const assistantNodeIds = [];
    let lastReconstructedNodeId = null;
    for (const entry of [...unacked].sort((a, b) =>
        a.source_id < b.source_id ? -1 : a.source_id > b.source_id ? 1 : 0,
    )) {
        // Resolve parent: prefer a just-reconstructed entry, fall back to the tree.
        const parentNodeId =
            sourceIdToNodeId.get(entry.parent_source_id) ?? null;
        const newNodeId = createNodeFn("assistant", entry.content, parentNodeId);
        const newNode = messageTree.get(newNodeId);
        if (newNode) {
            newNode.source_id = entry.source_id;
            newNode.snapshot_uuid = snapshotUuid;
        }
        sourceIdToNodeId.set(entry.source_id, newNodeId);
        assistantNodeIds.push(newNodeId);
        lastReconstructedNodeId = newNodeId;
    }

    if (composeMode === "send-from-leaf") {
        // Reparent the pending user node under the last reconstructed assistant —
        // bidirectionally: detach from old parent's children, attach to new.
        if (lastReconstructedNodeId) {
            _reparent(messageTree, pendingUserNodeId, lastReconstructedNodeId);
        }
        return { assistantNodeIds, draftContent: null };
    }

    // edit / regenerate: detach the pending node from its parent's `children`,
    // then drop it from the tree. Return its content as a draft for restore.
    const pending = messageTree.get(pendingUserNodeId);
    const draftContent = pending ? pending.content : null;
    if (pending) {
        _detach(messageTree, pendingUserNodeId);
        messageTree.delete(pendingUserNodeId);
    }
    return { assistantNodeIds, draftContent };
}

function _detach(messageTree, nodeId) {
    const node = messageTree.get(nodeId);
    if (!node) {
        return;
    }
    const parent = messageTree.get(node.parentId);
    if (!parent || !Array.isArray(parent.children)) {
        return;
    }
    parent.children = parent.children.filter((id) => id !== nodeId);
    if (parent.activeChildId === nodeId) {
        // Fall back to the next-most-recent child, or clear if no siblings remain.
        parent.activeChildId =
            parent.children.length > 0
                ? parent.children[parent.children.length - 1]
                : null;
    }
}

function _reparent(messageTree, nodeId, newParentId) {
    const node = messageTree.get(nodeId);
    if (!node) {
        return;
    }
    _detach(messageTree, nodeId);
    node.parentId = newParentId;
    const newParent = messageTree.get(newParentId);
    if (newParent && Array.isArray(newParent.children)) {
        if (!newParent.children.includes(nodeId)) {
            newParent.children.push(nodeId);
        }
        newParent.activeChildId = nodeId;
    }
}

/**
 * Build the wire-format messages array from the active path of `messageTree`.
 *
 * Each message carries `{role, content}` plus `source_id` if the node has been
 * server-stamped. Newly-authored nodes (edits, regenerations, just-typed text)
 * omit `source_id` so the syncer assigns one.
 */
export function buildRequestMessages(messageTree, activePath) {
    return activePath.map((nodeId) => {
        const node = messageTree.get(nodeId);
        const msg = { role: node.role, content: node.content };
        if (node.source_id) {
            msg.source_id = node.source_id;
        }
        return msg;
    });
}

function _indexBySourceId(messageTree) {
    const map = new Map();
    for (const [nodeId, node] of messageTree.entries()) {
        if (node && node.source_id) {
            map.set(node.source_id, nodeId);
        }
    }
    return map;
}
