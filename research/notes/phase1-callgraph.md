# Phase 1: PSI event → kill call graph

Textual call graph from a kernel PSI threshold breach to a reaped victim.
Each arrow is annotated with the `file:line` where the transition is
encoded.

```
kernel raises EPOLLPRI on PSI fd
  fd was opened by init_psi_monitor                       (libpsi/psi.cpp:36)
  fd was added to epoll with epev.data.ptr =
      &vmpressure_hinfo[level] via register_psi_monitor   (libpsi/psi.cpp:86)
      → epoll_ctl(EPOLL_CTL_ADD, ..., EPOLLPRI)           (libpsi/psi.cpp:92)
  handler pointer installed:
      vmpressure_hinfo[level].handler = mp_event_psi      (lmkd.cpp:3393)
  per-level setup is driven by init_psi_monitors          (lmkd.cpp:3561)
      invoked from the reinit path                        (lmkd.cpp:3699)
  threshold values come from
      psi_partial_stall_ms / psi_complete_stall_ms read   (lmkd.cpp:4153)

│
▼ main epoll loop wakes
mainloop()                                                 (lmkd.cpp:~3946)
  → epoll_wait returns the PSI event                      (lmkd.cpp:3980)
       (alt. waits: lmkd.cpp:3995, lmkd.cpp:4007)
  → first pass skips this event (non-EPOLLHUP)            (lmkd.cpp:4025)
  → second pass picks it up                               (lmkd.cpp:4040)
  → handler_info = (event_handler_info*) evt->data.ptr    (lmkd.cpp:4049)
  → call_handler(handler_info, &poll_params, evt->events) (lmkd.cpp:4050)

│
▼ dispatch
call_handler()                                            (lmkd.cpp:3909)
  → handler_info->handler(handler_info->data, events,
                          poll_params)                    (lmkd.cpp:3915)
       resolves to mp_event_psi for PSI levels

│
▼ PSI shim
mp_event_psi(int data, uint32_t events,
             struct polling_params *poll_params)          (lmkd.cpp:3117)
  → packs level into union psi_event_data
  → __mp_event_psi(PSI, event_data, events, poll_params)  (lmkd.cpp:3119)

│
▼ kill decision
__mp_event_psi(enum event_source source,
               union psi_event_data data,
               uint32_t events,
               struct polling_params *poll_params)        (lmkd.cpp:2713)
  → (reads PSI/meminfo/vmstat, thrashing & watermark
     checks, determines min_score_adj)
  → find_and_kill_process(min_score_adj, &ki, &mi,
                          &wi, &curr_tm, &psi_data)       (lmkd.cpp:3075)
       (sibling kill sites in same function:
        lmkd.cpp:3314 and lmkd.cpp:3338)

│
▼ victim selection
find_and_kill_process()                                   (lmkd.cpp:2539)
  → iterates OOM_SCORE_ADJ_MAX down to min_score_adj      (lmkd.cpp:2546)
  → picks heaviest or tail-most proc                      (lmkd.cpp:2558)
  → kill_one_process(procp, min_score_adj, ki, mi, wi,
                     tm, pd)                              (lmkd.cpp:2564)

│
▼ per-victim kill
kill_one_process()                                        (lmkd.cpp:2422)
  → reaper.kill({ pidfd, pid, uid }, false)               (lmkd.cpp:2483)
       (second reaper.kill at lmkd.cpp:2297 belongs to the
        kill_done_handler retry path, not the PSI path)

│
▼ async reap (separate thread, set up at boot)
Reaper::init(int comm_fd)                                 (reaper.cpp:158)
  spawns thread pool running reaper_main                  (reaper.cpp:91)
  reaper_main loop:
    → dequeue_request()                                   (reaper.cpp:107)
    → pidfd_send_signal(target.pidfd, SIGKILL, ...)       (reaper.cpp:113)
    → process_mrelease(target.pidfd, 0)                   (reaper.cpp:123)
    → request_complete()                                  (reaper.cpp:135)
  comm_fd half (reaper_comm_fd[0]) is itself epoll-added
  back into the main loop via                             (lmkd.cpp:3768)
  so kill-completion is delivered as kill_done_handler.
```

## Notes

- Phase 0 reference card facts re-confirmed:
  `mp_event_psi` at `lmkd.cpp:3117`, `find_and_kill_process` at
  `lmkd.cpp:2539`, `init_psi_monitor` at `libpsi/psi.cpp:36`,
  `init_psi_monitors` at `lmkd.cpp:3561` (Phase-0 card said 3579 — the
  function declaration line in *this* tree is 3561; the
  `psi_partial_stall_ms` / `psi_complete_stall_ms` *kernel-threshold
  application* lives inside its body and is the spirit of the Phase-0
  pointer).
- `Reaper::init` at `reaper.cpp:158` and `reaper_main` at `reaper.cpp:91`
  match the Phase-0 reference card.
- UNVERIFIED — exact opening line of the `mainloop()` function; the
  three `epoll_wait` call sites are inside its body
  (`lmkd.cpp:3980/3995/4007`) but I did not pin the function header
  line precisely. The dispatch claim itself is fully verified by the
  `call_handler` invocation at `lmkd.cpp:4050`.
