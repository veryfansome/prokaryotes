def _is_positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


def _range_is_valid(action: str, payload: dict, line_count: int) -> bool:
    start = payload.get("start_line")
    end = payload.get("end_line")
    if action == "insert_lines":
        if not _is_positive_int(start):
            return False
        # start_line in [1, line_count + 1]; end_line is unused.
        return 1 <= start <= line_count + 1
    if action in ("replace_lines", "delete_lines"):
        if not _is_positive_int(start) or not _is_positive_int(end):
            return False
        if line_count == 0:
            return False
        return 1 <= start <= end <= line_count
    return False


def _read_end_line(payload: dict, start_line: int) -> int | None:
    end_line = payload.get("end_line")
    if end_line is None:
        return None
    if isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < 1:
        raise ValueError("end_line for read_lines must be null or an integer >= 1")
    if end_line < start_line:
        raise ValueError("end_line for read_lines must be >= start_line")
    return end_line


def _read_start_line(payload: dict) -> int:
    start_line = payload.get("start_line")
    if start_line is None:
        return 1
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        raise ValueError("start_line for read_lines must be null or an integer >= 1")
    return start_line


def _validate_create_payload(payload: dict) -> str | None:
    if payload.get("expected_revision") is not None:
        return "expected_revision must be null for create_file."
    if payload.get("start_line") is not None:
        return "start_line must be null for create_file."
    if payload.get("end_line") is not None:
        return "end_line must be null for create_file."
    new_text = payload.get("new_text")
    if not isinstance(new_text, str):
        return "new_text is required for create_file and must be a string."
    return None


def _validate_write_payload(action: str, payload: dict) -> str | None:
    start_line = payload.get("start_line")
    if not _is_positive_int(start_line):
        return f"start_line is required for {action} and must be an integer >= 1."
    if action in ("replace_lines", "delete_lines"):
        end_line = payload.get("end_line")
        if not _is_positive_int(end_line):
            return f"end_line is required for {action} and must be an integer >= 1."
        if start_line > end_line:
            return f"start_line must be <= end_line for {action}."
    if action in ("replace_lines", "insert_lines"):
        new_text = payload.get("new_text")
        if not isinstance(new_text, str) or new_text == "":
            return f"new_text is required for {action} and must be a non-empty string."
    return None
