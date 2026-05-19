import { describe, it, expect } from "vitest";
import {
    applyBotMessage,
    applyHandshake,
    applyResyncHandshake,
    buildRequestMessages,
    relabelSnapshotUuid,
} from "../../scripts/static/conversation_client.js";

// messageTree is a `Map<nodeId, node>` to match the production data structure
// in `scripts/static/ui.js` (`const messageTree = new Map()`). Nodes also
// carry `children: number[]` and `activeChildId: id | null` so the
// bidirectional-link invariants `applyResyncHandshake` maintains are testable.
function makeTree(nodes) {
    const tree = new Map();
    for (const node of nodes) {
        if (!Array.isArray(node.children)) {
            node.children = [];
        }
        if (!("activeChildId" in node)) {
            node.activeChildId = null;
        }
        tree.set(node.id, node);
    }
    // Wire children arrays from parentId back-references so tests don't have
    // to spell out both sides of every link.
    for (const node of tree.values()) {
        if (node.parentId != null) {
            const parent = tree.get(node.parentId);
            if (parent && !parent.children.includes(node.id)) {
                parent.children.push(node.id);
                parent.activeChildId = node.id;
            }
        }
    }
    return tree;
}

function makeNode(id, role, content, parentId = null) {
    return { id, role, content, parentId, children: [], activeChildId: null };
}

function fakeCreateNode(messageTree) {
    let counter = 100;
    // Mirror production `createMessageNode`: wire `children` / `activeChildId`
    // on the parent and initialize empties on the new node.
    return (role, content, parentId) => {
        const newId = `n${++counter}`;
        messageTree.set(newId, {
            id: newId,
            role,
            content,
            parentId,
            children: [],
            activeChildId: null,
        });
        const parent = messageTree.get(parentId);
        if (parent && Array.isArray(parent.children)) {
            parent.children.push(newId);
            parent.activeChildId = newId;
        }
        return newId;
    };
}

describe("applyHandshake", () => {
    it("stamps assigned source_ids on the right indices", () => {
        const messageTree = makeTree([
            makeNode("n1", "user", "first"),
            makeNode("n2", "user", "second"),
        ]);
        const sentClientIds = ["n1", "n2"];
        const handshake = {
            snapshot_uuid: "s-A",
            source_id_assignments: [
                { client_index: 0, source_id: "1717000000.111" },
                { client_index: 1, source_id: "1717000000.222" },
            ],
        };
        const count = applyHandshake(messageTree, sentClientIds, handshake);
        expect(count).toBe(2);
        expect(messageTree.get("n1").source_id).toBe("1717000000.111");
        expect(messageTree.get("n1").snapshot_uuid).toBe("s-A");
        expect(messageTree.get("n2").source_id).toBe("1717000000.222");
        expect(messageTree.get("n2").snapshot_uuid).toBe("s-A");
    });

    it("preserves existing source_id but always restamps snapshot_uuid", () => {
        // Divergence: client sent a node with an existing source_id but the
        // server created a new branch. The handshake's snapshot_uuid is
        // authoritative even for nodes that already had source_ids.
        const messageTree = makeTree([makeNode("n1", "user", "edited")]);
        messageTree.get("n1").source_id = "1716999000.0";
        messageTree.get("n1").snapshot_uuid = "s-old";

        applyHandshake(messageTree, ["n1"], {
            snapshot_uuid: "s-new-branch",
            source_id_assignments: [],
        });

        expect(messageTree.get("n1").source_id).toBe("1716999000.0");
        expect(messageTree.get("n1").snapshot_uuid).toBe("s-new-branch");
    });

    it("stamps snapshot_uuid even on nodes without assignments", () => {
        // Invariant: every submitted user node gets snapshot_uuid from the handshake,
        // even if it already had a server-assigned source_id from a prior turn.
        const messageTree = makeTree([makeNode("n1", "user", "follow-up")]);
        applyHandshake(messageTree, ["n1"], {
            snapshot_uuid: "s-A",
            source_id_assignments: [],
        });
        expect(messageTree.get("n1").snapshot_uuid).toBe("s-A");
        expect(messageTree.get("n1").source_id).toBeUndefined();
    });

    it("returns 0 when handshake has no snapshot_uuid", () => {
        const count = applyHandshake(new Map(), [], { snapshot_uuid: null });
        expect(count).toBe(0);
    });
});

describe("applyBotMessage", () => {
    it("creates assistant node with both ids stamped", () => {
        const messageTree = makeTree([makeNode("n1", "user", "ask")]);
        const createNodeFn = fakeCreateNode(messageTree);
        const newId = applyBotMessage(messageTree, {
            parentNodeId: "n1",
            fullResponse: "answer",
            snapshotUuid: "s-A",
            sourceId: "1717000000.999",
            createNodeFn,
        });
        const node = messageTree.get(newId);
        expect(node.role).toBe("assistant");
        expect(node.content).toBe("answer");
        expect(node.parentId).toBe("n1");
        expect(node.source_id).toBe("1717000000.999");
        expect(node.snapshot_uuid).toBe("s-A");
    });
});

describe("relabelSnapshotUuid", () => {
    it("idempotent: updates only matching nodes", () => {
        const messageTree = makeTree([
            { ...makeNode("n1", "user", "a"), snapshot_uuid: "s-old" },
            { ...makeNode("n2", "assistant", "b"), snapshot_uuid: "s-old" },
            { ...makeNode("n3", "user", "c"), snapshot_uuid: "s-sibling" },
        ]);
        const firstPass = relabelSnapshotUuid(messageTree, "s-old", "s-new");
        expect(firstPass).toBe(2);
        expect(messageTree.get("n1").snapshot_uuid).toBe("s-new");
        expect(messageTree.get("n2").snapshot_uuid).toBe("s-new");
        expect(messageTree.get("n3").snapshot_uuid).toBe("s-sibling");

        // Second pass is a no-op
        const secondPass = relabelSnapshotUuid(messageTree, "s-old", "s-new");
        expect(secondPass).toBe(0);
    });

    it("does not touch sibling branches", () => {
        const messageTree = makeTree([
            { ...makeNode("n1", "user", "a"), snapshot_uuid: "s-branch-1" },
            { ...makeNode("n2", "user", "b"), snapshot_uuid: "s-branch-2" },
        ]);
        relabelSnapshotUuid(messageTree, "s-branch-1", "s-branch-1-compacted");
        expect(messageTree.get("n1").snapshot_uuid).toBe("s-branch-1-compacted");
        expect(messageTree.get("n2").snapshot_uuid).toBe("s-branch-2");
    });

    it("returns 0 on no-op input (null or same uuids)", () => {
        expect(relabelSnapshotUuid(new Map(), null, "x")).toBe(0);
        expect(relabelSnapshotUuid(new Map(), "x", "x")).toBe(0);
    });
});

describe("applyResyncHandshake — send-from-leaf", () => {
    it("reparents pending user under last reconstructed assistant (bidirectional)", () => {
        // Scenario: client previously acked u1@1.000001 and b1@1.000002. Just typed u2
        // locally as n3 (no source_id yet). Server-stored has trailing b2@1.000003 (a
        // multi-post continuation or follow-up bot) that the client missed.
        const messageTree = makeTree([
            { id: "n1", role: "user", content: "u1", parentId: null, source_id: "1.000001", snapshot_uuid: "s-A" },
            { id: "n2", role: "assistant", content: "b1", parentId: "n1", source_id: "1.000002", snapshot_uuid: "s-A" },
            { id: "n3", role: "user", content: "u2 (pending — server hasn't seen it)", parentId: "n2" },
        ]);
        const createNodeFn = fakeCreateNode(messageTree);

        const result = applyResyncHandshake(messageTree, {
            pendingUserNodeId: "n3",
            handshake: {
                snapshot_uuid: "s-A",
                source_id_assignments: [],
                unacknowledged_bot_messages: [
                    { source_id: "1.000003", content: "missed bot reply", parent_source_id: "1.000002" },
                ],
            },
            composeMode: "send-from-leaf",
            createNodeFn,
        });

        expect(result.assistantNodeIds).toHaveLength(1);
        expect(result.draftContent).toBeNull();
        const reconstructedId = result.assistantNodeIds[0];
        const reconstructed = messageTree.get(reconstructedId);
        expect(reconstructed.role).toBe("assistant");
        expect(reconstructed.content).toBe("missed bot reply");
        expect(reconstructed.source_id).toBe("1.000003");
        expect(reconstructed.snapshot_uuid).toBe("s-A");
        // The reconstructed bot is parented under the node whose source_id matches
        // its parent_source_id — that's n2 (b1), not n1 (u1).
        expect(reconstructed.parentId).toBe("n2");
        // ...and n2's children/activeChildId include the reconstructed bot.
        expect(messageTree.get("n2").children).toContain(reconstructedId);

        // Pending user n3 was detached from its old parent (n2) and reparented
        // under the reconstructed bot — both sides of the link.
        expect(messageTree.get("n3").parentId).toBe(reconstructedId);
        expect(messageTree.get("n2").children).not.toContain("n3");
        expect(messageTree.get(reconstructedId).children).toContain("n3");
        expect(messageTree.get(reconstructedId).activeChildId).toBe("n3");
    });

    it("chained bots reconstruct as a chain (bidirectional)", () => {
        const messageTree = makeTree([
            { id: "n1", role: "user", content: "u1", parentId: null, source_id: "1.0", snapshot_uuid: "s-A" },
            { id: "n2", role: "user", content: "u2 pending", parentId: "n1" },
        ]);
        const createNodeFn = fakeCreateNode(messageTree);
        const result = applyResyncHandshake(messageTree, {
            pendingUserNodeId: "n2",
            handshake: {
                snapshot_uuid: "s-A",
                unacknowledged_bot_messages: [
                    { source_id: "1.1", content: "b-a", parent_source_id: "1.0" },
                    { source_id: "1.2", content: "b-b", parent_source_id: "1.1" },
                ],
            },
            composeMode: "send-from-leaf",
            createNodeFn,
        });

        expect(result.assistantNodeIds).toHaveLength(2);
        const [firstId, secondId] = result.assistantNodeIds;
        expect(messageTree.get(firstId).parentId).toBe("n1");
        expect(messageTree.get(secondId).parentId).toBe(firstId); // chained
        expect(messageTree.get("n2").parentId).toBe(secondId); // reparented under last
        // children arrays follow the chain.
        expect(messageTree.get("n1").children).toContain(firstId);
        expect(messageTree.get(firstId).children).toContain(secondId);
        expect(messageTree.get(secondId).children).toContain("n2");
        // The pending node's old parent (n1) no longer references it.
        expect(messageTree.get("n1").children).not.toContain("n2");
    });
});

describe("applyResyncHandshake — edit / regenerate", () => {
    it("pops pending user node, detaches from parent.children, returns draft", () => {
        const messageTree = makeTree([
            { id: "n1", role: "user", content: "u1", parentId: null, source_id: "1.0", snapshot_uuid: "s-A" },
            { id: "n2", role: "assistant", content: "b1", parentId: "n1", source_id: "1.1", snapshot_uuid: "s-A" },
            { id: "n3", role: "user", content: "u2 typed via edit", parentId: "n2" },
        ]);
        const createNodeFn = fakeCreateNode(messageTree);
        const result = applyResyncHandshake(messageTree, {
            pendingUserNodeId: "n3",
            handshake: {
                snapshot_uuid: "s-A",
                unacknowledged_bot_messages: [
                    { source_id: "1.2", content: "missed bot", parent_source_id: "1.1" },
                ],
            },
            composeMode: "edit",
            createNodeFn,
        });

        expect(result.draftContent).toBe("u2 typed via edit");
        expect(messageTree.has("n3")).toBe(false); // popped from the Map
        // ...and removed from its parent's children list — no dangling reference.
        expect(messageTree.get("n2").children).not.toContain("n3");
        // Parent's activeChildId no longer points at the popped node.
        expect(messageTree.get("n2").activeChildId).not.toBe("n3");
        expect(result.assistantNodeIds).toHaveLength(1);
        const reconstructed = messageTree.get(result.assistantNodeIds[0]);
        expect(reconstructed.content).toBe("missed bot");
    });
});

describe("buildRequestMessages", () => {
    it("emits source_id only for server-stamped nodes", () => {
        const messageTree = makeTree([
            { id: "n1", role: "user", content: "u1", source_id: "1.0" },
            { id: "n2", role: "assistant", content: "b1", source_id: "1.1" },
            { id: "n3", role: "user", content: "fresh user input" }, // no source_id
        ]);
        const msgs = buildRequestMessages(messageTree, ["n1", "n2", "n3"]);
        expect(msgs).toEqual([
            { role: "user", content: "u1", source_id: "1.0" },
            { role: "assistant", content: "b1", source_id: "1.1" },
            { role: "user", content: "fresh user input" },
        ]);
    });
});
