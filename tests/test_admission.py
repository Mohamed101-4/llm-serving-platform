from llmserve.router.admission import AdmissionController
from llmserve.router.policies import Worker


def warmed(ctrl: AdmissionController, w: Worker, service_s=1.0, ttft_s=0.05) -> Worker:
    ctrl.record(w, latency_s=service_s, ttft_s=ttft_s, queued=False)
    return w


def test_admits_everything_before_any_observations():
    ctrl = AdmissionController(slo_ttft_ms=100)
    w = Worker("a", inflight=1000)
    assert ctrl.decide(w, [w]).admitted


def test_free_slot_estimate_is_base_ttft():
    ctrl = AdmissionController(slo_ttft_ms=500, default_slots=4)
    w = warmed(ctrl, Worker("a", inflight=3))
    assert ctrl.queue_ahead(w) == -1
    assert ctrl.estimate_ttft_ms(w) == 50


def test_queue_wait_grows_with_queue_length():
    ctrl = AdmissionController(slo_ttft_ms=500, default_slots=4)
    w = warmed(ctrl, Worker("a"))
    w.inflight = 4  # all slots busy: next request waits for 1 of 4 to finish
    assert ctrl.estimate_ttft_ms(w) == 1.0 / 4 * 1000 + 50
    w.inflight = 11  # 7 ahead in queue
    assert ctrl.estimate_ttft_ms(w) == 8 * 1.0 / 4 * 1000 + 50


def test_rejects_with_retry_after_when_every_worker_breaks_slo():
    ctrl = AdmissionController(slo_ttft_ms=500, default_slots=4)
    a = warmed(ctrl, Worker("a", inflight=20))
    b = warmed(ctrl, Worker("b", inflight=20))
    d = ctrl.decide(a, [a, b])
    assert not d.admitted
    assert d.retry_after_s >= 1
    assert ctrl.rejected == 1


def test_redirects_to_worker_within_slo():
    ctrl = AdmissionController(slo_ttft_ms=500, default_slots=4)
    busy = warmed(ctrl, Worker("busy", inflight=20))
    idle = warmed(ctrl, Worker("idle", inflight=0))
    d = ctrl.decide(busy, [busy, idle])
    assert d.worker is idle


def test_learns_slots_from_saturated_worker():
    ctrl = AdmissionController(slo_ttft_ms=500, default_slots=16)
    w = Worker("a", metrics={"running": 6, "waiting": 3})
    assert ctrl.slots(w) == 6
    w.metrics = {"running": 2, "waiting": 0}  # not saturated: keep the learned value
    assert ctrl.slots(w) == 6


def test_queued_requests_dont_inflate_base_ttft():
    ctrl = AdmissionController(slo_ttft_ms=500)
    w = warmed(ctrl, Worker("a"), ttft_s=0.05)
    ctrl.record(w, latency_s=1.0, ttft_s=5.0, queued=True)
    assert ctrl.stats(w).base_ttft_s == 0.05
