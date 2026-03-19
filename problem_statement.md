# ML Model Serving Infrastructure
### Local Simulation — Project Specification

---

## Problem Statement

ML models have no standardized path from training to a monitored, reliably-served API. Teams manually deploy, can't roll back safely, and have no visibility into degradation until users complain. You're building the infrastructure layer that solves this — entirely on local hardware.

**Goal:** `git push` → tested → built → deployed → monitored → auto-recoverable. No cloud required.

---

## What's Realistic Locally

| Production Thing | Local Equivalent |
|---|---|
| EC2 cluster | 3 VMs via Vagrant or multipass |
| EKS / managed k8s | k3s across those VMs |
| ECR (image registry) | Local `registry:2` container |
| AWS Secrets Manager | HashiCorp Vault (local) |
| Cloud load balancer | MetalLB + nginx ingress |

---

## Phases

### Phase 1 — Provision the Cluster
**Terraform + Ansible + Vagrant**
- Terraform provisions 3 local VMs via Vagrant provider
- Ansible configures: k3s install, firewall (ufw), TLS certs (cert-manager), non-root users
- > **Production touch:** Network policies blocking all pod-to-pod traffic by default, whitelist only

**Goal:** `terraform apply` → working k3s cluster from scratch, repeatable every time

---

### Phase 2 — Model Serving
**Docker + Triton Inference Server**
- Export a small HuggingFace model (e.g. `distilbert-base`) to ONNX
- Write Triton model repository layout, Dockerfile, Helm chart
- Deploy with liveness probe (HTTP) + readiness gate that waits for model load
- > **Production touch:** Non-root container, read-only filesystem, resource limits set

**Goal:** `curl localhost/v2/models/distilbert/infer` returns a real prediction

---

### Phase 3 — CI/CD Pipeline
**GitHub Actions + local registry**
- Pipeline: lint → unit test → `docker build` → Trivy image scan (fail on CRITICAL CVEs) → push to local registry → `kubectl rollout`
- Separate workflows for model changes vs infra changes
- > **Production touch:** Canary deploy — 10% traffic to new version via ingress weight, auto-rollback if p95 latency exceeds threshold (checked via Prometheus query in the pipeline)

**Goal:** merge a PR → new model version live in < 3 min, zero downtime

---

### Phase 4 — Security
**Vault + mTLS + Trivy**
- Vault running in k8s, injecting secrets as env vars via sidecar — no base64 k8s Secrets
- mTLS between all services using cert-manager + self-signed CA
- Signed images with cosign — admission webhook rejects unsigned images
- > **Production touch:** This entire layer is what most tutorials skip

**Goal:** `kubectl exec` into a pod and prove it can't reach another pod without explicit policy

---

### Phase 5 — Observability
**Prometheus + Grafana + OpenTelemetry**
- Prometheus scrapes Triton's `/metrics` — requests/sec, queue depth, p95 latency
- Grafana dashboard with SLO burn rate alerts (not threshold alerts)
- OpenTelemetry tracing: full trace from HTTP request → Triton → model backend
- > **Production touch:** Each Grafana alert links to a runbook markdown file in your repo

**Goal:** kill a pod mid-request, show the trace captures exactly where it failed

---

### Phase 6 — Resilience
**Chaos Mesh + HPA + PDB**
- HPA scaling on custom metric (Triton queue depth via Prometheus adapter) — not CPU
- PodDisruptionBudget: always keep ≥ 1 replica during rolling updates
- Chaos Mesh: schedule random pod kills, prove p99 stays under SLO
- > **Production touch:** Write a real postmortem doc from something that actually broke

**Goal:** `chaos-mesh` kills 1 of 3 pods → service survives → HPA scales back up → document it

---

## Benchmarks

Run `locust` against your endpoint and capture:
- Requests/sec at steady state
- p95 / p99 inference latency
- Time from `git push` to live (canary + full rollout)
- Recovery time after chaos pod kill