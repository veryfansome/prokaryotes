/**
 * UI-layer wire-parsing tests.
 *
 * Targets `scripts/static/ui.js`:
 * - First event is a `handshake` with `snapshot_uuid` + `source_id_assignments`.
 * - `bot_message` marks the final commit with the bot's server-assigned `source_id`.
 * - Other events (`context_pct`, `text_delta`, `progress_message`, `tool_call`, `compaction_pending`) keep their
 *   shape.
 *
 * Pure protocol primitives are covered by `conversation_client.test.js`. The DOM/fetch flow inside `createChatApp`
 * is covered by `chat_protocol_integration.test.js` (POST body shape, handshake stamping, bot_message gating,
 * resync recovery, error display, compaction polling).
 */
import { describe, it, expect } from "vitest";
import { parseStreamPayloadLine } from "../../scripts/static/ui.js";

describe("parseStreamPayloadLine — handshake", () => {
    it("parses a handshake with snapshot_uuid and source_id_assignments", () => {
        const parsed = parseStreamPayloadLine(
            JSON.stringify({
                snapshot_uuid: "s-A",
                source_id_assignments: [
                    { client_index: 0, source_id: "1.000001" },
                ],
            }),
        );
        expect(parsed).toEqual({
            type: "handshake",
            snapshot_uuid: "s-A",
            source_id_assignments: [{ client_index: 0, source_id: "1.000001" }],
        });
    });

    it("preserves unacknowledged_bot_messages on resync handshake", () => {
        const parsed = parseStreamPayloadLine(
            JSON.stringify({
                snapshot_uuid: "s-A",
                source_id_assignments: [],
                unacknowledged_bot_messages: [
                    { source_id: "1.000002", content: "missed", parent_source_id: "1.000001" },
                ],
            }),
        );
        expect(parsed.type).toBe("handshake");
        expect(parsed.unacknowledged_bot_messages).toHaveLength(1);
        expect(parsed.unacknowledged_bot_messages[0].source_id).toBe("1.000002");
    });

    it("rejects malformed handshake (non-string snapshot_uuid)", () => {
        expect(() =>
            parseStreamPayloadLine(
                JSON.stringify({ snapshot_uuid: 42, source_id_assignments: [] }),
            ),
        ).toThrow(/snapshot_uuid must be a string/);
    });

    it("rejects malformed handshake (non-array source_id_assignments)", () => {
        expect(() =>
            parseStreamPayloadLine(
                JSON.stringify({ snapshot_uuid: "s-A", source_id_assignments: "nope" }),
            ),
        ).toThrow(/source_id_assignments must be an array/);
    });
});

describe("parseStreamPayloadLine — bot_message", () => {
    it("parses a bot_message with its server-assigned source_id", () => {
        const parsed = parseStreamPayloadLine(
            JSON.stringify({ bot_message: { source_id: "1.000999" } }),
        );
        expect(parsed).toEqual({ type: "bot_message", source_id: "1.000999" });
    });

    it("rejects malformed bot_message (missing source_id)", () => {
        expect(() =>
            parseStreamPayloadLine(JSON.stringify({ bot_message: {} })),
        ).toThrow(/bot_message.source_id must be a string/);
    });
});

describe("parseStreamPayloadLine — other event shapes survive", () => {
    it.each([
        ["text_delta", { text_delta: "hi" }, { type: "text_delta", text_delta: "hi" }],
        [
            "progress_message",
            { progress_message: "Checking…" },
            { type: "progress_message", progress_message: "Checking…" },
        ],
        [
            "tool_call",
            { tool_call: { name: "shell_command", arguments: "{}" } },
            { type: "tool_call", tool_call: { name: "shell_command", arguments: "{}" } },
        ],
        ["context_pct", { context_pct: 42 }, { type: "context_pct", context_pct: 42 }],
        ["compaction_pending", { compaction_pending: true }, { type: "compaction_pending" }],
    ])("%s", (_label, payload, expected) => {
        expect(parseStreamPayloadLine(JSON.stringify(payload))).toEqual(expected);
    });
});
