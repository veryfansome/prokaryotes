import { describe, it, expect } from "vitest";
import {
    applyBotMessage,
    applyHandshake,
    applyResyncHandshake,
    buildRequestMessages,
    relabelSnapshotUuid,
} from "../../scripts/static/conversation_client.js";

function makeNode(id, role, content, parentId = null) {
    return { id, role, content, parentId };
}

function fakeCreateNode(messageTree) {
    let counter = 100;
    return (role, content, parentId) => {
        const newId = `n${++counter}`;
        messageTree[newId] = { id: newId, role, content, parentId };
        return newId;
    };
}

describe("applyHandshake", () => {
    it("stamps assigned source_ids on the right indices", () => {
        const messageTree = {
            n1: makeNode("n1", "user", "first"),
            n2: makeNode("n2", "user", "second"),
        };
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
        expect(messageTree.n1.source_id).toBe("1717000000.111");
        expect(messageTree.n1.snapshot_uuid).toBe("s-A");
        expect(messageTree.n2.source_id).toBe("1717000000.222");
        expect(messageTree.n2.snapshot_uuid).toBe("s-A");
    });

    it("preserves existing source_id but always restamps snapshot_uuid", () => {
        // Divergence: client sent a node with an existing source_id but the
        // server created a new branch. The handshake's snapshot_uuid is
        // authoritative even for nodes that already had source_ids.
        const messageTree = {
            n1: makeNode("n1", "user", "edited"),
        };
        messageTree.n1.source_id = "1716999000.0";
        messageTree.n1.snapshot_uuid = "s-old";

        applyHandshake(messageTree, ["n1"], {
            snapshot_uuid: "s-new-branch",
            source_id_assignments: [],
        });

        expect(messageTree.n1.source_id).toBe("1716999000.0");
        expect(messageTree.n1.snapshot_uuid).toBe("s-new-branch");
    });

    it("stamps snapshot_uuid even on nodes without assignments", () => {
        // Invariant: every submitted user node gets snapshot_uuid from the handshake,
        // even if it already had a server-assigned source_id from a prior turn.
        const messageTree = { n1: makeNode("n1", "user", "follow-up") };
        applyHandshake(messageTree, ["n1"], {
            snapshot_uuid: "s-A",
            source_id_assignments: [],
        });
        expect(messageTree.n1.snapshot_uuid).toBe("s-A");
        expect(messageTree.n1.source_id).toBeUndefined();
    });

    it("returns 0 when handshake has no snapshot_uuid", () => {
        const count = applyHandshake({}, [], { snapshot_uuid: null });
        expect(count).toBe(0);
    });
});

describe("applyBotMessage", () => {
    it("creates assistant node with both ids stamped", () => {
        const messageTree = { n1: makeNode("n1", "user", "ask") };
        const createNodeFn = fakeCreateNode(messageTree);
        const newId = applyBotMessage(messageTree, {
            parentNodeId: "n1",
            fullResponse: "answer",
            snapshotUuid: "s-A",
            sourceId: "1717000000.999",
            createNodeFn,
        });
        const node = messageTree[newId];
        expect(node.role).toBe("assistant");
        expect(node.content).toBe("answer");
        expect(node.parentId).toBe("n1");
        expect(node.source_id).toBe("1717000000.999");
        expect(node.snapshot_uuid).toBe("s-A");
    });
});

describe("relabelSnapshotUuid", () => {
    it("idempotent: updates only matching nodes", () => {
        const messageTree = {
            n1: { ...makeNode("n1", "user", "a"), snapshot_uuid: "s-old" },
            n2: { ...makeNode("n2", "assistant", "b"), snapshot_uuid: "s-old" },
            n3: { ...makeNode("n3", "user", "c"), snapshot_uuid: "s-sibling" },
        };
        const firstPass = relabelSnapshotUuid(messageTree, "s-old", "s-new");
        expect(firstPass).toBe(2);
        expect(messageTree.n1.snapshot_uuid).toBe("s-new");
        expect(messageTree.n2.snapshot_uuid).toBe("s-new");
        expect(messageTree.n3.snapshot_uuid).toBe("s-sibling");

        // Second pass is a no-op
        const secondPass = relabelSnapshotUuid(messageTree, "s-old", "s-new");
        expect(secondPass).toBe(0);
    });

    it("does not touch sibling branches", () => {
        const messageTree = {
            n1: { ...makeNode("n1", "user", "a"), snapshot_uuid: "s-branch-1" },
            n2: { ...makeNode("n2", "user", "b"), snapshot_uuid: "s-branch-2" },
        };
        relabelSnapshotUuid(messageTree, "s-branch-1", "s-branch-1-compacted");
        expect(messageTree.n1.snapshot_uuid).toBe("s-branch-1-compacted");
        expect(messageTree.n2.snapshot_uuid).toBe("s-branch-2");
    });

    it("returns 0 on no-op input (null or same uuids)", () => {
        expect(relabelSnapshotUuid({}, null, "x")).toBe(0);
        expect(relabelSnapshotUuid({}, "x", "x")).toBe(0);
    });
});

describe("applyResyncHandshake — send-from-leaf", () => {
    it("reparents pending user under last reconstructed assistant + auto-retry expected", () => {
        // Scenario: client previously acked u1@1.000001 and b1@1.000002. Just typed u2
        // locally as n3 (no source_id yet). Server-stored has trailing b2@1.000003 (a
        // multi-post continuation or follow-up bot) that the client missed.
        const messageTree = {
            n1: { id: "n1", role: "user", content: "u1", parentId: null, source_id: "1.000001", snapshot_uuid: "s-A" },
            n2: { id: "n2", role: "assistant", content: "b1", parentId: "n1", source_id: "1.000002", snapshot_uuid: "s-A" },
            n3: { id: "n3", role: "user", content: "u2 (pending — server hasn't seen it)", parentId: "n2" },
        };
        const createNodeFn = fakeCreateNode(messageTree);
        const activeChildUpdates = [];
        const setActiveChildFn = (parentId, childId) => activeChildUpdates.push([parentId, childId]);

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
            setActiveChildFn,
        });

        expect(result.assistantNodeIds).toHaveLength(1);
        expect(result.draftContent).toBeNull();
        const reconstructed = messageTree[result.assistantNodeIds[0]];
        expect(reconstructed.role).toBe("assistant");
        expect(reconstructed.content).toBe("missed bot reply");
        expect(reconstructed.source_id).toBe("1.000003");
        expect(reconstructed.snapshot_uuid).toBe("s-A");
        // The reconstructed bot is parented under the node whose source_id matches
        // its parent_source_id — that's n2 (b1), not n1 (u1).
        expect(reconstructed.parentId).toBe("n2");

        // Pending user node n3 is reparented under the reconstructed bot
        expect(messageTree.n3.parentId).toBe(result.assistantNodeIds[0]);
        // activeChildId was updated to keep n3 as the leaf
        expect(activeChildUpdates).toEqual([[result.assistantNodeIds[0], "n3"]]);
    });

    it("chained bots reconstruct as a chain", () => {
        const messageTree = {
            n1: { id: "n1", role: "user", content: "u1", parentId: null, source_id: "1.0", snapshot_uuid: "s-A" },
            n2: { id: "n2", role: "user", content: "u2 pending", parentId: "n1" },
        };
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
            setActiveChildFn: () => {},
        });

        expect(result.assistantNodeIds).toHaveLength(2);
        const [firstId, secondId] = result.assistantNodeIds;
        expect(messageTree[firstId].parentId).toBe("n1");
        expect(messageTree[secondId].parentId).toBe(firstId);  // chained
        expect(messageTree.n2.parentId).toBe(secondId);  // reparented under last
    });
});

describe("applyResyncHandshake — edit / regenerate", () => {
    it("pops pending user node, returns draft content, no reparenting", () => {
        const messageTree = {
            n1: { id: "n1", role: "user", content: "u1", parentId: null, source_id: "1.0", snapshot_uuid: "s-A" },
            n2: { id: "n2", role: "assistant", content: "b1", parentId: "n1", source_id: "1.1", snapshot_uuid: "s-A" },
            n3: { id: "n3", role: "user", content: "u2 typed via edit", parentId: "n2" },
        };
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
            setActiveChildFn: () => {},
        });

        expect(result.draftContent).toBe("u2 typed via edit");
        expect(messageTree.n3).toBeUndefined();  // popped
        expect(result.assistantNodeIds).toHaveLength(1);
        const reconstructed = messageTree[result.assistantNodeIds[0]];
        expect(reconstructed.content).toBe("missed bot");
    });
});

describe("buildRequestMessages", () => {
    it("emits source_id only for server-stamped nodes", () => {
        const messageTree = {
            n1: { role: "user", content: "u1", source_id: "1.0" },
            n2: { role: "assistant", content: "b1", source_id: "1.1" },
            n3: { role: "user", content: "fresh user input" }, // no source_id
        };
        const msgs = buildRequestMessages(messageTree, ["n1", "n2", "n3"]);
        expect(msgs).toEqual([
            { role: "user", content: "u1", source_id: "1.0" },
            { role: "assistant", content: "b1", source_id: "1.1" },
            { role: "user", content: "fresh user input" },
        ]);
    });
});
