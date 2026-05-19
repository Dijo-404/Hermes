# Phase 1: PSI epoll wiring

## Dispatch wiring (the answer)

The PSI file descriptor is connected to `mp_event_psi` in two coordinated
steps inside `psi_monitor_enable` (the per-level monitor setup helper):

1. **Handler installation:** `lmkd.cpp:3393`

   ```
   vmpressure_hinfo[level].handler = use_new_strategy ? mp_event_psi : mp_event_common;
   vmpressure_hinfo[level].data = level;
   ```

   The `struct event_handler_info` slot for this pressure level
   (`vmpressure_hinfo[]` is declared at `lmkd.cpp:301`) gets its function
   pointer set to `mp_event_psi` when the "new strategy" is in effect.

2. **epoll registration:** `lmkd.cpp:3395`

   ```
   if (register_psi_monitor(epollfd, fd, &vmpressure_hinfo[level]) < 0) {
   ```

   `register_psi_monitor` is defined in `libpsi/psi.cpp:86` and performs the
   actual `epoll_ctl(epollfd, EPOLL_CTL_ADD, fd, &epev)` at
   `libpsi/psi.cpp:92`, with `epev.events = EPOLLPRI` (line 90) and
   `epev.data.ptr = data` (line 91, i.e. the address of the
   `vmpressure_hinfo[level]` slot). This is the single epoll-add call that
   wires the PSI fd into the main loop; `plan.md`'s reference to
   `poll_kernel` was incorrect — that path (`lmkd.cpp:846`) is the eBPF
   kill-outcome ring buffer reader, not the PSI dispatch.

## How a kernel PSI event traverses the code

1. The kernel raises `EPOLLPRI` on the PSI fd previously opened by
   `init_psi_monitor` (`libpsi/psi.cpp:36`) and registered above
   (`libpsi/psi.cpp:86`).
2. The main loop's `epoll_wait` call returns that event. There are three
   `epoll_wait` call sites depending on the polling state of the loop:
   `lmkd.cpp:3980` (timed wait during a polling window), `lmkd.cpp:3995`
   (kill-timeout-bounded wait), and `lmkd.cpp:4007` (blocking wait when
   idle).
3. The loop iterates events in a two-pass scan: the first pass at
   `lmkd.cpp:4025` handles `EPOLLHUP` (data-socket disconnects and
   `kill_done_handler`); the second pass at `lmkd.cpp:4040` handles
   everything else. For a PSI fd readiness the event is non-HUP, so the
   second pass takes it.
4. At `lmkd.cpp:4048-4050`, the loop reads `evt->data.ptr` back as an
   `event_handler_info*` (this is the same pointer that
   `register_psi_monitor` stored in `epev.data.ptr` at
   `libpsi/psi.cpp:91`) and forwards it to `call_handler`.
5. `call_handler` is defined at `lmkd.cpp:3909`; at line 3915 it invokes
   `handler_info->handler(handler_info->data, events, poll_params)`. For
   PSI levels wired in step 1 that resolves to `mp_event_psi`.
6. `mp_event_psi` at `lmkd.cpp:3117` is a thin shim: it packs the
   pressure-level integer into a `union psi_event_data` and delegates to
   `__mp_event_psi(PSI, event_data, events, poll_params)` at
   `lmkd.cpp:3119`. `__mp_event_psi` is defined at `lmkd.cpp:2713` and
   carries the actual kill-decision logic.

## Secondary entry into the same hot path

There is a *second* invocation site for `__mp_event_psi` at
`lmkd.cpp:3485` that is reached from the vendor `memevent_listener`
path rather than from the PSI epoll dispatch — relevant context for
Phase 2 because it shares the kill-decision code but bypasses the PSI
`event_handler_info` indirection.

## Notes / caveats

- The handler-pointer at line 3393 is conditional on `use_new_strategy`.
  When false, the same epoll slot routes to `mp_event_common` instead;
  `mp_event_psi` is therefore only reached when the new-strategy code
  path is selected. (`lmkd.cpp:3393`)
- `vmpressure_hinfo[]` is also reused by the memcg-style registration at
  `lmkd.cpp:3649-3651`, where it is hard-wired to `mp_event_common` and
  stashed into `epev.data.ptr` by an `epoll_ctl(EPOLL_CTL_ADD)` at
  `lmkd.cpp:3652`. That is a *different* event source (memory cgroup
  eventfd) — it shares the dispatch table but does not feed
  `mp_event_psi`.
- `init_psi_monitors` (the plural; `lmkd.cpp:3561`) is the orchestrator
  that calls the per-level setup containing line 3393/3395. It is invoked
  during the reinit path at `lmkd.cpp:3699`.
