# AgentIQ on OpenShift

Manifests for running the AgentIQ API on OpenShift (tested on the Red Hat
Developer Sandbox, OpenShift 4.21). The `k8s/` directory one level up targets
vanilla Kubernetes and is kept as a reference; these are the ones that
actually deploy here.

## What differs from `k8s/`

| `k8s/` (vanilla)                     | `k8s/openshift/`                                   | Why |
|-------------------------------------|---------------------------------------------------|-----|
| nginx `Ingress`, hardcoded host     | `Route`, edge TLS, router-assigned host            | OpenShift has no nginx ingress controller |
| `Namespace` object + `namespace:`   | none                                               | The sandbox doesn't allow creating namespaces |
| `securityContext.runAsUser: 1000`   | `runAsNonRoot` only                                | `restricted-v2` assigns a UID from the namespace range; a fixed UID fails admission |
| `ReadWriteMany` PVC for the index   | none — index baked into the image                  | The default storage class is `ReadWriteOnce`; can't back multiple replicas |
| `image: agentiq:1.0.0`              | in-cluster registry ref, built by `build.yaml`     | No prebuilt image; on-cluster build avoids arch mismatch |
| HPA on CPU + memory                 | HPA on CPU only, `minReplicas: 1`                  | Resident model size sits near a memory target at idle |

## Deploy

```sh
# 1. build the image on-cluster from the working tree
oc apply -f k8s/openshift/build.yaml
oc start-build agentiq --from-dir=. --follow

# 2. API keys (OpenAI required, Tavily optional)
oc create secret generic agentiq-secrets \
  --from-literal=OPENAI_API_KEY=... \
  --from-literal=TAVILY_API_KEY=...

# 3. deploy
oc apply -f k8s/openshift/deployment.yaml \
         -f k8s/openshift/service.yaml \
         -f k8s/openshift/hpa.yaml

oc rollout status deploy/agentiq-api
oc get route agentiq-api -o jsonpath='{.spec.host}'
```

## Notes

- `AGENTIQ_API_KEY` is not set, so `/chat` is unauthenticated. Fine for a
  short-lived demo on an unlisted route; set it (and add it to
  `agentiq-secrets`) for anything longer-lived.
- The `build.yaml` `resources` block is load-bearing — see the comment in
  that file.
- The Developer Sandbox idles workloads after a period of inactivity; the
  first request after an idle period pays a cold start.
