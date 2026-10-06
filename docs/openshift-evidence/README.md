# OpenShift deployment evidence

This folder is a point-in-time record of the AgentIQ deployment as it existed on
a live OpenShift cluster. It was captured because the cluster account was about
to be deactivated and everything in it deleted.

**Captured on:** 2026-10-05

**Cluster:** Red Hat Developer Sandbox (free, time-limited), OpenShift
`api.rm2.thpm.p1.openshiftapps.com`

**Namespace:** `saduvamsikrishna513-dev`

**Status at capture:** the sandbox trial was 3 days from expiry. The namespace
and all objects below were deleted when it expired, so the Route URL in these
files no longer works and cannot be revived. The manifests in `k8s/openshift/`
remain reproducible on any OpenShift cluster.

## What the captured files show

| File | Contents |
|---|---|
| `01-get-all-wide.txt` | Deployment, Services, Route and HPA with wide output |
| `02-describe-hpa.txt` | Full HPA state, limits and conditions |
| `03-describe-deploy.txt` | Full Deployment state, probes, resources and env sources |
| `04-events.txt` | Namespace events at capture time (empty, see below) |
| `05-builds-imagestreams.txt` | Builds, BuildConfig, ImageStream and ImageStreamTag |
| `06-describe-build.txt` | The on-cluster build that produced the running image |
| `07-route-and-health.txt` | Route hostname and the response from `/health` |
| `live-manifests/` | The objects as they actually existed on the cluster, as YAML |

## Key facts these files record

- The **Route existed and was served by the cluster router** with edge TLS and an
  HTTP to HTTPS redirect, at
  `agentiq-api-saduvamsikrishna513-dev.apps.rm2.thpm.p1.openshiftapps.com`.
- The **image was built on the cluster**, not pushed from a laptop. Build
  `agentiq-3` completed in 7m56s and produced image digest
  `sha256:7639a0a6...`, tagged `agentiq:latest` in the namespace ImageStream.
- That build was made from **git commit `77c2ccd`** ("add OpenShift deployment
  manifests"), which ties the image that ran on the cluster to a specific commit
  in this repository.
- The **HPA was configured and attached** to the Deployment, targeting 60% CPU
  with `minReplicas: 1` and `maxReplicas: 6`.
- The Deployment carried the **full probe set** (startup, readiness, liveness,
  all on `/health`), resource requests and limits, and `runAsNonRoot: true`.

## Important limits on this evidence

These are stated so the files are not read as showing more than they do.

- **No pods were running at capture time.** The Deployment was scaled to 0
  replicas because the Developer Sandbox idles inactive workloads. The HPA
  therefore reports `cpu: <unknown>` and `ScalingActive: False` with reason
  `ScalingDisabled`. The pod list in `01-get-all-wide.txt` is empty for the
  same reason.
- **The `/health` request returned HTTP 503.** This is the cluster router
  answering with its standard error page because there was no pod behind the
  Service. It shows the Route was live and routing; it does not show the
  application responding.
- **The namespace event history was already empty.** OpenShift expires events
  after about an hour, and the load test ran weeks before this capture. The
  HPA scale-up and scale-down events are not recoverable and are not in this
  folder.
- **The load test Job no longer existed**, so neither the Job object nor its
  pod logs could be saved. `k8s/openshift/loadtest-job.yaml` is the manifest
  that was used.
- Nothing in this folder was produced by scaling, restarting or otherwise
  changing the cluster. Every command was read-only.

## Redactions

- `live-manifests/buildconfig.yaml`: the GitHub and generic **webhook trigger
  secrets** were replaced with `REDACTED`. These are auto-generated values that
  allow triggering a build, and they are not needed to understand the
  configuration.
- No Secret objects were exported. The application reads its credentials from a
  Secret named `agentiq-secrets` via `envFrom`, so only that name appears in
  these files, never any value.
