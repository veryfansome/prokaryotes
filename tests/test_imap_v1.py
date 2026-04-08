from types import SimpleNamespace

import pytest

import prokaryotes.imap_v1 as imap_v1


class FakeStopEvent:
    def __init__(self, set_after_wait_calls: int | None = None):
        self._is_set = False
        self._set_after_wait_calls = set_after_wait_calls
        self.wait_calls: list[float] = []

    def is_set(self) -> bool:
        return self._is_set

    def set(self):
        self._is_set = True

    def wait(self, timeout: float) -> bool:
        self.wait_calls.append(timeout)
        if self._is_set:
            return True
        if self._set_after_wait_calls and len(self.wait_calls) >= self._set_after_wait_calls:
            self._is_set = True
            return True
        return False


class FakeIdleClient:
    def __init__(self, stop_event: FakeStopEvent | None = None, stop_after_idle_checks: int | None = None):
        self.idle_calls = 0
        self.idle_check_calls = 0
        self.idle_done_calls = 0
        self.selected_folders: list[str] = []
        self.stop_event = stop_event
        self.stop_after_idle_checks = stop_after_idle_checks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def select_folder(self, folder: str):
        self.selected_folders.append(folder)

    def idle(self):
        self.idle_calls += 1

    def idle_check(self, timeout: int):
        del timeout
        self.idle_check_calls += 1
        if self.stop_event and self.stop_after_idle_checks and self.idle_check_calls >= self.stop_after_idle_checks:
            self.stop_event.set()
        return []

    def idle_done(self):
        self.idle_done_calls += 1
        return "OK", []


def make_controller_for_idle_loop_test(fake_client: FakeIdleClient, fake_event: FakeStopEvent):
    controller = imap_v1.IngestController.__new__(imap_v1.IngestController)
    controller.folder = "INBOX"
    controller.idle_manager_client = None
    controller.imap_client_factory = SimpleNamespace(get_client=lambda: fake_client)
    controller.stop_event = fake_event
    return controller


@pytest.mark.parametrize("input_tuple, expected_dict", [
    # Standard case: multiple pairs
    (
        (b'CHARSET', b'UTF-8', b'NAME', b'file.txt'),
        {'charset': 'UTF-8', 'name': 'file.txt'}
    ),
    # Edge case: Empty input
    (None, {}),
    ((), {}),
])
def test_parse_params(input_tuple, expected_dict):
    assert imap_v1.parse_params(input_tuple) == expected_dict


def test_run_idle_manager_backs_off_on_rapid_empty_checks():
    fake_event = FakeStopEvent(set_after_wait_calls=1)
    fake_client = FakeIdleClient()
    controller = make_controller_for_idle_loop_test(fake_client, fake_event)

    controller.run_idle_manager(
        idle_check_spin_backoff_seconds=0.01,
        idle_check_spin_threshold_seconds=5,
        idle_check_timeout=60,
    )

    assert fake_client.idle_check_calls == 1
    assert fake_event.wait_calls == [0.01]


def test_run_idle_manager_refreshes_idle_after_repeated_rapid_empty_checks():
    fake_event = FakeStopEvent()
    fake_client = FakeIdleClient(stop_event=fake_event, stop_after_idle_checks=4)
    controller = make_controller_for_idle_loop_test(fake_client, fake_event)

    controller.run_idle_manager(
        idle_check_spin_backoff_seconds=0.01,
        idle_check_spin_restart_threshold=3,
        idle_check_spin_threshold_seconds=5,
        idle_check_timeout=60,
    )

    assert fake_client.idle_done_calls == 1
    assert fake_client.idle_calls == 2  # initial IDLE + one refresh
