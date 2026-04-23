# Kubernetes Architecture Deep Dive — docs-agent on OCI

Everything you need to understand and explain about how the docs-agent system lives on a Kubernetes cluster. Read this top-to-bottom — each section builds on the previous one.

---

## 1. The Physical Layer: What Is the Cluster?

A Kubernetes cluster is **a set of machines (nodes) that run your containerized applications**, managed by a control plane that decides what runs where.

Your proposal uses **OKE (Oracle Kubernetes Engine)** — Oracle's managed Kubernetes service. "Managed" means Oracle runs the control plane (the API server, etcd, scheduler, controller-manager) for you. You only manage the **worker nodes** where your actual pods run.

### Your Two Node Pools

From your Terraform config (section 3.5):

```
┌─────────────────────────────────────────────────────────────┐
│                    OKE Cluster (v1.29)                       │
│                                                             │
│  ┌───────────── CPU Node Pool ─────────────┐                │
│  │  3 × VM.Standard.E4.Flex               │                │
│  │  4 OCPUs, 32 GB RAM each               │                │
│  │                                         │                │
│  │  Runs:                                  │                │
│  │   • API Server pod                      │                │
│  │   • MCP Server pod                      │                │
│  │   • Milvus (vector DB) pod              │                │
│  │   • Frontend pod                        │                │
│  │   • Redis pod                           │                │
│  │   • Istio control plane                 │                │
│  │   • KFP (pipeline) controllers          │                │
│  │   • Kagent controller                   │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
│  ┌───────────── GPU Node Pool ─────────────┐                │
│  │  1 × VM.GPU.A10.1                      │                │
│  │  NVIDIA A10 GPU                         │                │
│  │                                         │                │
│  │  TAINTED: nvidia.com/gpu=present:NoSchedule              │
│  │                                         │                │
│  │  Runs ONLY:                             │                │
│  │   • KServe InferenceService (Llama 3.1) │                │
│  └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Why the Taint Matters

A **taint** is a "keep away" sign on a node. The GPU node has:
```yaml
taints:
  - key: "nvidia.com/gpu"
    value: "present"
    effect: "NoSchedule"
```

This means: **no pod can be scheduled on this node unless it explicitly tolerates this taint**. Without this, Kubernetes might schedule your Redis pod or your frontend on the expensive GPU node, wasting GPU resources.

The KServe InferenceService for Llama has a matching **toleration** (either explicitly or via KServe defaults for GPU workloads), so it's the only thing that runs there.

**Why 3 CPU nodes?** High availability. If one node dies, Kubernetes automatically reschedules pods to the other two. With 1 node, a node failure = total downtime.

---

## 2. Kubernetes Fundamentals: Pods, Services, Deployments

### Pod
The smallest deployable unit. A pod is **one or more containers** that share the same network namespace (same IP, same localhost). Your MCP server is one pod. Your API server is one pod. But with Istio, each pod actually has **two containers**: your application container + the Envoy sidecar (more on this later).

### Deployment
A Deployment says "I want N replicas of this pod running at all times." If a pod dies, the Deployment controller creates a new one.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: docs-agent
spec:
  replicas: 2         # ← "I want 2 copies of this pod"
  selector:
    matchLabels:
      app: api-server
  template:
    spec:
      containers:
        - name: api-server
          image: ghcr.io/kubeflow/docs-agent-api:latest
          ports:
            - containerPort: 8080
```

### Service
A Service gives a **stable DNS name and IP** to a set of pods. Pods are ephemeral — they get new IPs every time they restart. A Service is permanent.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: api-server
  namespace: docs-agent
spec:
  selector:
    app: api-server      # ← "route traffic to pods with this label"
  ports:
    - port: 8080
      targetPort: 8080
```

This creates a DNS entry: `api-server.docs-agent.svc.cluster.local`

Any pod in the cluster can reach the API server using this DNS name. This is why your `values.yaml` has entries like:
```yaml
llm:
  endpoint: "http://llama.docs-agent.svc.cluster.local/openai/v1/chat/completions"
milvus:
  host: "milvus.docs-agent.svc.cluster.local"
```

These are **internal Kubernetes DNS names**, not public URLs. They only work inside the cluster.

### Namespace
A namespace is a **logical partition** of the cluster. Everything for docs-agent lives in the `docs-agent` namespace. This provides:
- **Isolation**: your pods don't interfere with other teams' pods
- **RBAC scoping**: permissions can be scoped to a namespace
- **Resource quotas**: you can limit how much CPU/memory the namespace can consume

---

## 3. What Are CRDs? (Custom Resource Definitions)

This is a fundamental Kubernetes concept that underpins KServe, Kagent, and Istio.

### The Standard Resources

Kubernetes ships with built-in resource types: `Pod`, `Service`, `Deployment`, `ConfigMap`, `Secret`, `Namespace`, etc. You interact with them via `kubectl`:

```bash
kubectl get pods -n docs-agent
kubectl get services -n docs-agent
```

### The Problem
What if you want to tell Kubernetes "I need an LLM inference service with this model, 1 GPU, auto-scaling to 2 replicas"? There's no built-in resource for that. Kubernetes doesn't know what an "InferenceService" is.

### The Solution: CRDs
A **CRD (Custom Resource Definition)** lets you **teach Kubernetes new resource types**. It extends the Kubernetes API.

When KServe is installed, it registers CRDs like:
```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: inferenceservices.serving.kserve.io
spec:
  group: serving.kserve.io
  names:
    kind: InferenceService
    plural: inferenceservices
    shortNames: ["isvc"]
```

After this CRD is registered, you can do:
```bash
kubectl get inferenceservices -n docs-agent
# or shorthand:
kubectl get isvc -n docs-agent
```

Kubernetes now "understands" what an InferenceService is — but it doesn't know what to **do** with it yet. That's where **controllers** come in.

### The Controller Pattern (CRD + Controller = Operator)

A **controller** is a program running in the cluster that watches for CRD instances and **reconciles reality to match the desired state**.

```
You create:
  InferenceService "llama" with model=Llama-3.1-8B, gpu=1

KServe controller sees this and:
  1. Creates a Deployment with the vLLM container
  2. Creates a Service pointing to it
  3. Creates a Knative Revision for auto-scaling
  4. Configures the Istio VirtualService for traffic routing
  5. Exposes an OpenAI-compatible /v1/chat/completions endpoint

You just wrote 20 lines of YAML.
The controller created 200+ lines of underlying resources for you.
```

**This is why CRDs are powerful**: you describe *what* you want at a high level, and the controller figures out *how* to make it happen.

---

## 4. The CRDs In Your System

Your cluster has CRDs from **four** different projects. Here's every CRD that matters:

### 4.1 KServe CRDs

| CRD | What you write | What the controller creates |
|-----|----------------|----------------------------|
| **ServingRuntime** | "Use this container image with these flags as a model server" | A reusable template for how to serve models |
| **InferenceService** | "Deploy model X using runtime Y with N GPUs" | Pod, Service, Knative Route, auto-scaler, health checks |

From your proposal (section 3.2):
```yaml
# ServingRuntime: defines HOW to run a model server
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: llm-runtime
spec:
  containers:
    - image: kserve/huggingfaceserver:v0.14.0-gpu  # pinned version!
      name: kserve-container
      args: ["--backend=vllm"]

# InferenceService: declares WHICH model to serve
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama
  namespace: docs-agent
spec:
  predictor:
    model:
      runtime: llm-runtime       # ← references the ServingRuntime above
      args:
        - --model_id=RedHatAI/Llama-3.1-8B-Instruct
        - --enable-auto-tool-choice    # ← vLLM native function calling
        - --tool-call-parser=llama3_json
      resources:
        requests:
          nvidia.com/gpu: "1"     # ← this triggers GPU scheduling
```

When you `kubectl apply` this, the KServe controller:
1. Creates a Deployment with the vLLM container on the GPU node (only the GPU node has the `nvidia.com/gpu` resource)
2. Downloads the model from HuggingFace using the `HF_TOKEN` secret
3. Creates an internal Service at `http://llama.docs-agent.svc.cluster.local`
4. Exposes OpenAI-compatible endpoints (`/openai/v1/chat/completions`)
5. Sets up auto-scaling: `minReplicas: 1`, `maxReplicas: 2`, `scaleTarget: 10`

**What vLLM is:** vLLM is the actual inference engine inside the container. It's a high-performance LLM serving library. The `--enable-auto-tool-choice` and `--tool-call-parser=llama3_json` flags enable the model to output structured function calls (tool calling), which is essential for the agent to invoke `search_kubeflow_docs`.

### 4.2 Kagent CRDs

Kagent is a Kubernetes-native agent framework. It has **four** CRDs that compose together:

```
ModelConfig ──► Agent ◄── RemoteMCPServer
                              │
                              └──► references the MCP Server pod
```

| CRD | Purpose | Your instance |
|-----|---------|---------------|
| **ModelConfig** | Points to an LLM endpoint | `kserve-llama` → points to the KServe InferenceService |
| **RemoteMCPServer** | Declares an external tool server | `kubeflow-docs-mcp` → points to the MCP Server pod |
| **Agent** | The top-level agent definition | `kubeflow-docs-agent` → ties ModelConfig + tools + system prompt together |
| **Tool** | Individual tool definitions | Not used directly — tools are declared inside the MCP server |

From your proposal (section 2.4):
```yaml
# Step 1: Tell Kagent where the LLM lives
apiVersion: kagent.dev/v1alpha2
kind: ModelConfig
metadata:
  name: kserve-llama
spec:
  provider: OpenAI                      # ← uses OpenAI-compatible API
  openAI:
    baseUrl: "http://llama.docs-agent.svc.cluster.local/openai/v1"
  model: llama3.1-8B

# Step 2: Tell Kagent where the tools live
apiVersion: kagent.dev/v1alpha2
kind: RemoteMCPServer         # ← "there's a tool server at this URL"
metadata:
  name: kubeflow-docs-mcp
spec:
  url: "http://mcp-kubeflow-docs.docs-agent.svc.cluster.local:8000/mcp"

# Step 3: Compose the agent
apiVersion: kagent.dev/v1alpha2
kind: Agent
metadata:
  name: kubeflow-docs-agent
spec:
  type: Declarative
  declarative:
    modelConfig: kserve-llama                       # ← which LLM
    tools:
      - type: McpServer
        mcpServer:
          name: kubeflow-docs-mcp                   # ← which tools
          toolNames: ["search_kubeflow_docs"]        # ← which specific tools
    systemMessage: |
      You are the Kubeflow Docs Assistant...         # ← behavior rules
```

**What the Kagent controller does:** When you apply these CRDs, the controller:
1. Reads the Agent CR, resolves the ModelConfig and RemoteMCPServer references
2. Starts a managed agent (likely a pod) that connects to the LLM endpoint and the MCP server
3. Handles the agent loop: receive query → call LLM → if tool_call → execute via MCP → feed result back to LLM → return final answer
4. Manages lifecycle: restarts, health checks, scaling — all via Kubernetes-native mechanisms

**Why `type: Declarative`?** This means "the Kagent controller manages the agent's lifecycle entirely." You don't write custom orchestration code. You just declare the desired state and the controller handles it. The alternative would be running your own custom agent loop in a container (Architecture A from the spec).

### 4.3 Istio CRDs

| CRD | Purpose |
|-----|---------|
| **Gateway** | Configures the ingress gateway to accept external traffic |
| **VirtualService** | Routes incoming traffic to backend services |
| **AuthorizationPolicy** | Access control: who can talk to whom |
| **PeerAuthentication** | Enforces mTLS between services |
| **DestinationRule** | Traffic policies (load balancing, circuit breaking) |

More on these in the Istio section below.

---

## 5. Istio Service Mesh: The Invisible Security Layer

### What Is Istio?

Istio is a **service mesh** — a dedicated infrastructure layer that controls how services communicate with each other. Think of it as a "smart network" between all of your pods.

### How It Works: The Sidecar Pattern

When Istio is enabled on a namespace, every pod gets an extra container injected automatically: an **Envoy proxy sidecar**.

```
┌─────────────── Pod: api-server ───────────────┐
│                                                │
│  ┌──────────────┐    ┌──────────────────────┐  │
│  │ Your app     │◄──►│ Envoy Sidecar        │  │
│  │ (FastAPI)    │    │ (injected by Istio)   │  │
│  │ port 8080    │    │ intercepts ALL        │  │
│  │              │    │ inbound + outbound    │  │
│  │              │    │ network traffic       │  │
│  └──────────────┘    └──────────────────────┘  │
│                              ▲                  │
│                              │                  │
└──────────────────────────────│──────────────────┘
                               │
                        Network traffic
                    never goes directly to
                    your app — always through
                    Envoy first
```

**Every single network packet** entering or leaving your application goes through this sidecar. Your application code doesn't know Istio exists. It just listens on port 8080 — Envoy intercepts the traffic transparently.

### What the Sidecar Does

1. **mTLS encryption** — encrypts all traffic between pods automatically
2. **AuthorizationPolicy enforcement** — blocks unauthorized traffic
3. **Telemetry collection** — emits metrics and traces for observability
4. **Retry logic** — can retry failed requests transparently
5. **Circuit breaking** — prevents cascading failures

### mTLS (Mutual TLS) — Why This Matters

Regular TLS (what HTTPS uses): the **client** verifies the **server's** identity. One-way.

**Mutual TLS**: **both sides** verify each other's identity using certificates. Your API server proves its identity to the MCP server, AND the MCP server proves its identity to the API server.

Istio does this automatically. When Istio is installed with `PeerAuthentication` set to `STRICT`:

```yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: docs-agent
spec:
  mtls:
    mode: STRICT     # ← ALL traffic must be mTLS. No plaintext allowed.
```

**Every** service-to-service call within the `docs-agent` namespace is encrypted. The important thing: **your application code uses plain HTTP** (`http://milvus.docs-agent.svc.cluster.local:19530`). The Envoy sidecars transparently upgrade this to mTLS. You don't write any TLS code.

```
API Server (HTTP) ──► Envoy A ═══[mTLS]═══► Envoy B ──► MCP Server (HTTP)
                      encrypts               decrypts
              Your app doesn't know this is happening
```

### AuthorizationPolicy — The Firewall

Your cluster has a **global deny-all policy** (section 3.4):

```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: global-deny-all
  namespace: docs-agent
spec: {}     # ← empty spec = deny everything
```

This means: **no pod can talk to any other pod** unless there's an explicit ALLOW policy. This is the zero-trust model.

Then you add specific ALLOW policies:

```yaml
# "The MCP server can only be reached by the API server and the Kagent controller"
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-mcp-server
spec:
  selector:
    matchLabels:
      app: mcp-kubeflow-docs          # ← applies to the MCP server pods
  action: ALLOW
  rules:
    - from:
        - source:
            principals:                # ← only these identities can call it
              - "cluster.local/ns/docs-agent/sa/api-server"
              - "cluster.local/ns/docs-agent/sa/kagent-controller"
      to:
        - operation:
            ports: ["8000"]            # ← only on this port
```

**`principals`** — these are **SPIFFE identities** derived from Kubernetes service accounts. When a pod runs with `serviceAccountName: api-server`, Istio issues it a certificate with the identity `cluster.local/ns/docs-agent/sa/api-server`. The AuthorizationPolicy checks this certificate.

This means: even if an attacker gets code execution inside the frontend pod, they **cannot** directly query the MCP server or Milvus, because the frontend's service account is not in the allow list.

### The Full AuthorizationPolicy Map

```
                    Who can talk to whom?

Frontend ──► API Server        ✅ (allowed)
Frontend ──► MCP Server        ❌ (blocked by deny-all)
Frontend ──► Milvus            ❌ (blocked)
Frontend ──► KServe LLM        ❌ (blocked)

API Server ──► MCP Server      ✅ (allowed, port 8000)
API Server ──► KServe LLM      ✅ (allowed, port 8080)
API Server ──► Milvus          ❌ (only MCP talks to Milvus)

MCP Server ──► Milvus          ✅ (allowed, port 19530)
MCP Server ──► KServe LLM      ❌ (MCP doesn't need the LLM)

Anything else ──► anything     ❌ (global deny-all)
```

---

## 6. Ingress: How External Traffic Enters the Cluster

### The Problem

Everything inside the cluster has internal DNS names (`*.svc.cluster.local`). These are **unreachable from the outside world**. How does a user's browser reach the frontend?

### The Solution: Istio Ingress Gateway

```
  Internet
     │
     │ HTTPS (port 443)
     ▼
┌─────────────────────────────┐
│  OCI Load Balancer          │ ← provisioned by Terraform
│  (public IP: 140.x.x.x)    │    (or automatically by Istio)
│  DNS: docs-agent.kubeflow.org
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Istio Ingress Gateway      │ ← a special Envoy pod at the cluster edge
│  (runs in istio-system ns)  │
│                             │
│  Does:                      │
│  • TLS termination          │
│  • Route based on hostname  │
│  • Set X-Forwarded-For      │
└────────────┬────────────────┘
             │
             ▼
     Internal cluster traffic (mTLS between sidecars)
```

### TLS Termination — Where HTTPS Ends

**TLS termination** means "this is where encrypted HTTPS traffic is decrypted." There are two common patterns:

**Pattern 1: Terminate at Load Balancer** — The OCI load balancer holds the TLS certificate. Traffic between the LB and the Ingress Gateway is plaintext (but it's within OCI's internal network).

**Pattern 2: Terminate at Ingress Gateway (preferred)** — The TLS certificate is stored as a Kubernetes Secret. The Ingress Gateway does the decryption. This is more secure because traffic is encrypted all the way to the cluster edge.

Your setup uses this via Istio Gateway + VirtualService CRDs:

```yaml
# Gateway: "accept HTTPS traffic for docs-agent.kubeflow.org"
apiVersion: networking.istio.io/v1
kind: Gateway
metadata:
  name: docs-agent-gateway
  namespace: docs-agent
spec:
  selector:
    istio: ingressgateway       # ← runs on the Istio ingress gateway pods
  servers:
    - port:
        number: 443
        name: https
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: docs-agent-tls-cert    # ← K8s Secret with TLS cert
      hosts:
        - "docs-agent.kubeflow.org"

# VirtualService: "route traffic to the right backend"
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: docs-agent-routes
spec:
  hosts:
    - "docs-agent.kubeflow.org"
  gateways:
    - docs-agent-gateway
  http:
    - match:
        - uri:
            prefix: /mcp        # ← IDE requests go to MCP server
      route:
        - destination:
            host: mcp-kubeflow-docs.docs-agent.svc.cluster.local
            port:
              number: 8000
    - match:
        - uri:
            prefix: /api        # ← chat UI API calls go to API server
      route:
        - destination:
            host: api-server.docs-agent.svc.cluster.local
            port:
              number: 8080
    - route:                     # ← everything else goes to frontend
        - destination:
            host: frontend.docs-agent.svc.cluster.local
            port:
              number: 3000
```

### The Full TLS/Encryption Stack

```
Browser ──── HTTPS ────► OCI LB ────► Istio Ingress Gateway
                                          │ TLS terminated here
                                          │ Plain HTTP internally
                                          ▼
                                     Frontend pod
                                          │
                                     (Envoy sidecar ←─── mTLS ───► Envoy sidecar)
                                          │
                                     API Server pod
                                          │
                                     (Envoy sidecar ←─── mTLS ───► Envoy sidecar)
                                          │
                                     MCP Server pod
                                          │
                                     (Envoy sidecar ←─── mTLS ───► Envoy sidecar)
                                          │
                                     Milvus pod
```

**Three layers of encryption:**
1. **HTTPS** — browser to ingress gateway (public internet)
2. **mTLS** — ingress gateway to first backend (within cluster)
3. **mTLS** — between all internal services (sidecar to sidecar)

---

## 7. DNS and Service Discovery

How does `llama.docs-agent.svc.cluster.local` actually resolve?

Kubernetes runs **CoreDNS** — a DNS server inside the cluster. Every Service automatically gets a DNS entry:

```
<service-name>.<namespace>.svc.cluster.local
```

When you create a Service named `llama` in namespace `docs-agent`, CoreDNS automatically registers:
- `llama.docs-agent.svc.cluster.local` → ClusterIP (e.g., `10.96.45.123`)

When your API server calls `http://llama.docs-agent.svc.cluster.local/openai/v1/chat/completions`:
1. CoreDNS resolves the hostname to the ClusterIP
2. kube-proxy routes the traffic to one of the healthy pods behind the Service
3. If there are 2 replicas, kube-proxy load-balances between them (round-robin)

**This is why internal URLs use `.svc.cluster.local`** — it's the DNS suffix that CoreDNS manages.

---

## 8. The Complete End-to-End Request Flow

This is what you need to be able to trace in the interview. A user asks "How do I configure autoscaling in Kubeflow Pipelines?" on the chat UI.

```
STEP 1: THE USER'S BROWSER
──────────────────────────
User types query in the chat UI at https://docs-agent.kubeflow.org
Frontend JavaScript sends: POST /api/chat {"query": "How do I configure autoscaling..."}


STEP 2: OCI LOAD BALANCER
──────────────────────────
Request hits OCI's public load balancer (140.x.x.x)
LB forwards to the Istio Ingress Gateway pod


STEP 3: ISTIO INGRESS GATEWAY
──────────────────────────────
Gateway terminates TLS (decrypts HTTPS)
VirtualService matches /api/* → routes to api-server.docs-agent.svc.cluster.local:8080
Sets X-Forwarded-For: <user's real IP>
Establishes mTLS connection to the API server's Envoy sidecar


STEP 4: API SERVER POD (Envoy sidecar)
───────────────────────────────────────
Envoy sidecar receives the request, verifies mTLS, passes to the FastAPI app on localhost:8080


STEP 5: API SERVER (FastAPI application)
────────────────────────────────────────
a) Rate Limit Middleware:
   - Extracts real IP from X-Forwarded-For
   - Checks Redis: INCR ratelimit:burst:<ip>:<minute> → under 5? OK
   - Checks Redis: INCR ratelimit:daily:<ip>:<date> → under 50? OK

b) Builds LLM request:
   - Constructs messages array with system prompt + user query
   - Includes tool definitions (search_kubeflow_docs schema)
   - Sends to: http://llama.docs-agent.svc.cluster.local/openai/v1/chat/completions


STEP 6: API SERVER → KSERVE LLM (first LLM call)
──────────────────────────────────────────────────
Traffic flow:
  API Server → Envoy A ═══[mTLS]═══► Envoy B → KServe vLLM container

The LLM receives the query and tool definitions.
It decides this is a docs question.
It returns a tool_call:
  {
    "tool_calls": [{
      "function": {
        "name": "search_kubeflow_docs",
        "arguments": {"query": "configure autoscaling pipelines", "source_filter": "docs"}
      }
    }]
  }


STEP 7: API SERVER → MCP SERVER (tool execution)
─────────────────────────────────────────────────
API Server sees tool_call, routes to MCP server:
  POST http://mcp-kubeflow-docs.docs-agent.svc.cluster.local:8000/mcp

Traffic flow:
  API Server → Envoy A ═══[mTLS]═══► Envoy C → MCP Server (FastMCP)


STEP 8: MCP SERVER (search execution)
──────────────────────────────────────
a) Embeds the query:
   - SentenceTransformer (all-mpnet-base-v2) encodes "configure autoscaling pipelines"
   - Produces a 768-dimension float vector

b) Searches Milvus:
   - Traffic: MCP Server → Envoy C ═══[mTLS]═══► Envoy D → Milvus pod
   - Query: ANN search on the vector, partition_names=["docs"], top_k=5
   - Returns 5 most similar document chunks with content_text + citation_url

c) Formats response:
   - Returns structured JSON: {"content": "...", "citations": ["https://kubeflow.org/..."]}


STEP 9: API SERVER (second LLM call)
─────────────────────────────────────
API Server receives tool results.
Constructs a new LLM message:
  - Original system prompt
  - User query
  - tool_call message from step 6
  - tool_result with the 5 retrieved chunks

Sends to KServe LLM again (same path as step 6).

The LLM now has real documentation context.
It generates a final answer grounded in the retrieved content.


STEP 10: RESPONSE STREAMS BACK
───────────────────────────────
LLM output streams back through:
  KServe → Envoy B → Envoy A → API Server → Envoy A → Istio Gateway → OCI LB → Browser

The frontend renders:
  - Action log: "Searched Kubeflow documentation ✓"
  - Citations: [blue links to kubeflow.org pages]
  - Formatted answer with the autoscaling configuration details
  - Feedback buttons (thumbs up/down)
```

### The Same Flow for IDE Users (MCP Direct)

The IDE flow is **shorter** — it skips the API server and the LLM entirely:

```
Developer IDE (Cursor)
    │
    │ HTTPS POST to https://docs-agent.kubeflow.org/mcp
    │ Header: Authorization: Bearer <API_KEY>
    ▼
OCI Load Balancer → Istio Ingress Gateway
    │ TLS termination
    │ VirtualService matches /mcp → routes to MCP server
    ▼
MCP Server
    │ API key middleware validates Bearer token
    │ Rate limit via Redis (per-key quota)
    │ Embeds query → Milvus search → returns results
    ▼
Returns thin context package:
  - Content snippet (150 tokens)
  - Validation URL (link to source)

Developer's LOCAL IDE agent (Claude/Copilot)
takes this context and generates the answer
using the developer's own LLM subscription.
```

**Why this is cost-effective:** For IDE users, our infrastructure only pays for the embedding + Milvus search. The expensive LLM generation happens on the developer's local machine using their own API credits.

---

## 9. KServe Deep Dive: How the LLM Actually Runs

### The Serving Stack

```
┌────────────── KServe InferenceService "llama" ──────────────┐
│                                                              │
│  ┌─────────┐     ┌──────────────────────────────────────┐   │
│  │ Envoy   │     │ kserve-container (vLLM)              │   │
│  │ sidecar │◄───►│                                      │   │
│  │         │     │ • Loads Llama-3.1-8B model weights    │   │
│  │         │     │ • Uses GPU memory (90% utilization)   │   │
│  │         │     │ • Max context: 32768 tokens           │   │
│  │         │     │ • Native tool calling enabled         │   │
│  │         │     │ • OpenAI-compatible API               │   │
│  │         │     │   POST /openai/v1/chat/completions    │   │
│  └─────────┘     └──────────────────────────────────────┘   │
│                                                              │
│  Scheduled on: GPU node (tolerates nvidia.com/gpu taint)     │
│  Resources: 4 CPU, 16Gi RAM, 1× NVIDIA A10 GPU              │
│  Auto-scaling: 1-2 replicas, target 10 concurrent requests   │
└──────────────────────────────────────────────────────────────┘
```

### Auto-Scaling Behavior

```yaml
minReplicas: 1    # always keep 1 warm (no cold start for first request)
maxReplicas: 2    # scale up to 2 under load
scaleTarget: 10   # target 10 concurrent requests per replica
```

- **0-10 concurrent requests:** 1 replica handles everything
- **11-20 concurrent requests:** KServe spins up a 2nd replica (takes ~30-60s for GPU model loading)
- **Back to <10:** After a cooldown period, the 2nd replica is terminated

`minReplicas: 1` is important — with `minReplicas: 0` (scale-to-zero), the first request after inactivity would wait 60+ seconds for the model to load into GPU memory. For a chat interface, that's unacceptable.

### Why vLLM Specifically?

vLLM provides:
- **PagedAttention** — efficient GPU memory management for concurrent requests
- **Continuous batching** — processes multiple requests simultaneously
- **OpenAI-compatible API** — your API server just makes standard HTTP calls
- **Native tool calling** — `--enable-auto-tool-choice` enables structured function call outputs

---

## 10. The Helm Chart: How It All Gets Deployed

Helm is a package manager for Kubernetes. Your Helm chart (`deployments/helm/docs-agent/`) contains templates that generate all the YAML manifests:

```
helm install docs-agent ./deployments/helm/docs-agent/ -n docs-agent
```

This single command creates:

| What | Kubernetes Resource | Template File |
|------|-------------------|---------------|
| Frontend UI | Deployment + Service | `frontend-deployment.yaml` |
| API Server | Deployment + Service | `api-server-deployment.yaml` |
| MCP Server | Deployment + Service | `mcp-server-deployment.yaml` |
| Redis | Deployment + Service | `redis-deployment.yaml` |
| Milvus | StatefulSet (via subchart) | Helm dependency |
| KServe ServingRuntime | ServingRuntime CR | `serving-runtime.yaml` |
| KServe InferenceService | InferenceService CR | `inference-service.yaml` |
| Kagent ModelConfig | ModelConfig CR | `kagent-model-config.yaml` |
| Kagent RemoteMCPServer | RemoteMCPServer CR | `kagent-mcp-server.yaml` |
| Kagent Agent | Agent CR | `kagent-agent.yaml` |
| Istio Gateway | Gateway CR | `istio-gateway.yaml` |
| Istio VirtualService | VirtualService CR | `istio-virtualservice.yaml` |
| Istio AuthorizationPolicies | AuthorizationPolicy CRs | `istio-authz-*.yaml` |
| Secrets | Secret | `secrets.yaml` (references external values) |

**Values templating** — the `values.yaml` file provides configurable knobs. When someone runs locally vs. OCI, they override:

```bash
# OCI deployment
helm install docs-agent ./chart/ -n docs-agent \
  -f values-oci.yaml    # storageClass: oci-bv, real GPU shapes

# Local Kind cluster
helm install docs-agent ./chart/ -n docs-agent \
  -f values-local.yaml  # storageClass: local-path, no GPU, maybe use API-based LLM
```

### Deployment Order Matters

Helm deploys everything simultaneously, but controllers reconcile asynchronously. The logical dependency order:

```
1. Secrets (credentials must exist first)
2. Milvus (vector DB must be up before MCP can connect)
3. Redis (rate limiter must be up before API server starts accepting traffic)
4. KServe ServingRuntime → InferenceService (LLM must be up for agent to work)
5. MCP Server (needs Milvus to be ready)
6. API Server (needs MCP + KServe to be ready)
7. Kagent CRDs (ModelConfig → RemoteMCPServer → Agent)
8. Frontend (needs API Server to be ready)
9. Istio routing (Gateway + VirtualService + AuthorizationPolicies)
```

In practice, Kubernetes handles this via **readiness probes** — each pod reports "I'm ready" only when its dependencies are confirmed reachable. Until then, the Service sends no traffic to it.

---

## 11. The Full Cluster Architecture Diagram

```
┌─────────────────────────────────── OKE Cluster ────────────────────────────────────┐
│                                                                                     │
│  ┌──────────── istio-system namespace ────────────┐                                 │
│  │                                                 │                                │
│  │  istiod (control plane)                         │                                │
│  │  istio-ingressgateway ◄──── OCI Load Balancer ◄──── Internet                     │
│  │                                                 │                                │
│  └─────────────────────────────────────────────────┘                                │
│                                                                                     │
│  ┌──────────── docs-agent namespace ──────────────────────────────────────────┐      │
│  │                                                                            │      │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │      │
│  │  │ Frontend     │──►│ API Server   │──►│ MCP Server   │──►│ Milvus     │  │      │
│  │  │ (React)      │   │ (FastAPI)    │   │ (FastMCP)    │   │ (Vector DB)│  │      │
│  │  │ port: 3000   │   │ port: 8080   │   │ port: 8000   │   │ port:19530 │  │      │
│  │  │ [+sidecar]   │   │ [+sidecar]   │   │ [+sidecar]   │   │ [+sidecar] │  │      │
│  │  └──────────────┘   └──────┬───────┘   └──────────────┘   └────────────┘  │      │
│  │                            │                                               │      │
│  │                            │            ┌──────────────┐                   │      │
│  │                            └───────────►│ KServe       │                   │      │
│  │                                         │ Llama 3.1-8B │                   │      │
│  │                            ┌────────────│ (vLLM+GPU)   │                   │      │
│  │                            │            │ port: 8080   │                   │      │
│  │                            │            │ [+sidecar]   │                   │      │
│  │                            │            └──────────────┘                   │      │
│  │  ┌──────────────┐          │                                               │      │
│  │  │ Redis        │◄─────────┘ (rate limit checks)                           │      │
│  │  │ port: 6379   │                                                          │      │
│  │  │ [+sidecar]   │        ┌─────────────────────────┐                       │      │
│  │  └──────────────┘        │ Kagent Controller       │                       │      │
│  │                          │ (watches Agent CRDs,    │                       │      │
│  │                          │  manages agent lifecycle)│                       │      │
│  │                          └─────────────────────────┘                       │      │
│  │                                                                            │      │
│  │  [+sidecar] = Envoy proxy injected by Istio, enforcing mTLS + AuthZ      │      │
│  └────────────────────────────────────────────────────────────────────────────┘      │
│                                                                                     │
│  ┌──────────── kubeflow namespace ────────────────┐                                 │
│  │  Kubeflow Pipelines (KFP) controller           │                                 │
│  │  (runs ingestion pipeline CronJobs)             │                                 │
│  └─────────────────────────────────────────────────┘                                 │
│                                                                                     │
│  ┌──────────── CPU Node Pool ──────────┐  ┌──── GPU Node Pool ────┐                 │
│  │  3 × VM.Standard.E4.Flex           │  │  1 × VM.GPU.A10.1    │                 │
│  │  All pods except LLM run here      │  │  ONLY KServe LLM     │                 │
│  └────────────────────────────────────┘  │  runs here (tainted)  │                 │
│                                           └───────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Mentor Q&A — Quick-Fire Answers

| Question | Your Answer |
|---|---|
| *"What's a CRD?"* | A Custom Resource Definition extends the Kubernetes API with new resource types. KServe adds `InferenceService`, Kagent adds `Agent`, Istio adds `AuthorizationPolicy`. Each comes with a controller that reconciles the desired state into concrete Kubernetes resources. |
| *"Why Istio?"* | Zero-trust network security via mTLS between all services, fine-grained access control via AuthorizationPolicies, and traffic management via VirtualServices — all without changing application code. The OCI cluster already runs Istio. |
| *"How does traffic enter the cluster?"* | HTTPS → OCI Load Balancer → Istio Ingress Gateway (TLS terminated) → VirtualService routes to backend Services → Envoy sidecars handle mTLS between pods. |
| *"What if the LLM pod crashes?"* | KServe's InferenceService has `minReplicas: 1`. The Deployment controller immediately creates a new pod. During the ~60s model loading time, requests queue at the Service. The API server's retry logic (`execute_tool_via_mcp`) handles transient failures. |
| *"What if Milvus crashes?"* | With Helm persistence enabled, data survives pod restarts. The MCP server's retry logic returns an explicit "unavailable" message. The LLM's system prompt instructs it to tell the user rather than hallucinate. |
| *"Why not put everything in one pod?"* | Separation of concerns + independent scaling. The LLM needs a GPU; Milvus needs disk IOPS; the frontend needs low latency. One pod can't optimize for all three. Independent Deployments let each component scale and restart independently. |
| *"How do you ensure the GPU pod only runs on the GPU node?"* | Node taint (`nvidia.com/gpu=present:NoSchedule`) + resource request (`nvidia.com/gpu: "1"`). The scheduler only places pods with GPU requests on nodes that have GPUs AND have matching tolerations. |
| *"What's the difference between the API server and the MCP server?"* | The API server handles user-facing HTTP/WebSocket connections and orchestrates LLM calls. The MCP server is a **tool server** — it only does vector search. This separation follows the MCP specification: the server exposes tools, clients consume them. One source of truth for retrieval logic. |
| *"Why Kagent instead of just running a FastAPI app?"* | Kagent manages the agent lifecycle as a Kubernetes-native resource. Health checks, restarts, scaling, and configuration are all handled by the Kagent controller via CRDs. You declare the desired agent state in YAML; the controller handles the operations. Changes are GitOps-friendly — you update YAML in git, ArgoCD syncs it. |
| *"What happens if a user sends 1000 requests per second?"* | Three layers of defense: (1) OCI Load Balancer can apply connection limits, (2) the application rate limiter in Redis blocks after burst limit (5/min) and daily limit (50/day), (3) KServe auto-scaler caps at maxReplicas=2 with scaleTarget=10, meaning max 20 concurrent LLM calls. Excess requests get HTTP 429 or queue. |
