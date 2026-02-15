
## Agent/Server Issues (Kagent Preparation)

### Issue 6: Modularize tool definitions for Kagent
**Labels**: `enhancement`, `proposal`, `kagent`

**Description**:
Currently, tools are defined as a Python list (`TOOLS`). Kagent uses Kubernetes CRDs to define tools.

**Proposed Solution**:
- Create a `tools/` directory with one file per tool (e.g., `search_kubeflow_docs.yaml`).
- Write a utility to load these definitions into both the current Python format and Kagent YAML.

**Why this helps Agentic RAG**:
This is the first concrete step towards Kagent integration.

---

### Issue 7: Add a "Reasoning" step before tool execution
**Labels**: `enhancement`, `proposal`

**Description**:
The current system directly passes the user's query to the LLM with tools enabled. An "agentic" system should first *reason* about the query to decide:
1. Does this require a tool?
2. Which tool(s) are needed?
3. How should the query be refined?

**Proposed Solution**:
Introduce a two-step LLM call:
1. **Planner Call**: Ask the LLM to output a structured plan (JSON).
2. **Executor Call**: Execute the plan and synthesize the response.

**Why this helps Agentic RAG**:
This is the core "agentic" loop that Kagent orchestrates. Implementing it manually first will clarify the requirements for Kagent integration.

---

## Infrastructure Issues (OCI Deployment)

### Issue 8: Add KServe `scaleToZero` configuration
**Labels**: `infrastructure`, `proposal`

**Description**:
The proposal mentions using KServe Scale-to-Zero for efficient handling of bursty workloads.

**Proposed Solution**:
Update `manifests/inference-service.yaml` with:
```yaml
spec:
  predictor:
    minReplicas: 0
    scaleTarget: 1
    scaleMetric: concurrency
```

**Why this helps Agentic RAG**:
This is a direct requirement of the proposal for OCI deployment.

---

### Issue 9: Create initial Terraform module for OCI/OKE
**Labels**: `infrastructure`, `proposal`

**Description**:
The proposal requires a reproducible Terraform deployment for OCI.

**Proposed Solution**:
Create `manifests/oci-reference-arch/` with:
- `main.tf`: Provider and VCN setup.
- `oke.tf`: OKE cluster definition.
- `variables.tf`: Configurable parameters.

**Why this helps Agentic RAG**:
This is the "Deployment Reference" deliverable in the proposal.
