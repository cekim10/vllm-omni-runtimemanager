# Deadline-Bounded Evacuation: Focused Literature Check

Checked before interpreting the offline evacuation kill test. This is a
focused novelty check, not a systematic literature review.

## Cloud notice windows

- AWS EC2 Spot normally provides a best-effort two-minute interruption
  notice: <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html>
- Azure Spot VMs provide best-effort scheduled-event delivery up to 30 seconds
  before eviction: <https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms>
- Google Cloud Spot VMs document configurable preemption notice durations,
  including a 120-second preview option; the default does not guarantee a
  dedicated delay: <https://docs.cloud.google.com/compute/docs/instances/spot>

These sources justify 30-second and 120-second primary deadlines. The
five-second deadline in the kill test is a stress case, not primary evidence.

## Closely related systems

- SpotServe serves generative LLMs on preemptible instances and adapts
  parallelization and migration to fluctuating capacity:
  <https://arxiv.org/abs/2311.15566>
- Parcae provides proactive migration and liveput optimization for DNN
  training on preemptible instances:
  <https://www.usenix.org/conference/nsdi24/presentation/duan>
- Can't Be Late studies scheduling spot jobs under deadlines and shows that a
  simple policy can close much of the optimal-policy gap:
  <https://www.usenix.org/conference/nsdi24/presentation/wu-zhanghao>
- ReclaimNet explicitly studies deadline-aware migration traffic control under
  provider reclaim notices:
  <https://arxiv.org/abs/2605.28872>
- GCR accelerates general GPU checkpoint/restore for elasticity, task
  switching, and fault tolerance:
  <https://www.usenix.org/conference/fast26/presentation/zeng>

## Novelty implication

Deadline-aware checkpointing, migration, and spot-resource scheduling are
already established systems problems. A video-diffusion paper therefore
cannot claim novelty from the reclaim deadline alone. It would need a measured
video-specific decision space, such as representation choices or session
interactions that leave a material gap over strong independent policies.

The current offline kill test does not find that gap: realistic cells can
evacuate all sessions, and even stressed cells leave less than 0.2% saved-work
headroom between value-density greedy and a fractional-oracle upper bound.
