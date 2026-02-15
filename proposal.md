Project 1: Agentic RAG on Kubeflow (Expansion of kubeflow/docs-agent)
Components: Kubeflow Pipelines (KFP), KServe, Manifests (Deployment/Infra), LLM Agents

Mentors: @chasecadet, @tarekabouzeid (Tarek Abouzeid - Kubeflow Platform)

Contributor: Details:

Project Overview & Scope: This project aims to evolve the existing kubeflow/docs-agent from a simple retrieval tool into a robust Reference Architecture for Agentic RAG on Kubeflow. Currently, the tool performs basic lookups. The GSoC contributor will upgrade this to an agentic workflow that can intelligently parse user questions, access the Kubeflow Git repository and Reference Platform Architecture as tools, and provide cited, technical answers. The core goal is "Dogfooding": We want to use Kubeflow to build the AI that helps users learn Kubeflow.

Key Deliverables (GSoC Scope):

Agentic Architecture: Implement an agent (using frameworks like LangGraph or Kagent) running on Kubeflow that can query specialized indices (Documentation, GitHub Issues, Platform Architecture).
Ingestion Pipelines: Build reusable Kubeflow Pipelines (KFP) to scrape, chunk, and index "Golden Data" from our reference architectures, establishing a best-practice pattern for data handling.
Local Serving via KServe: Demonstrate how to serve the agent's LLM (e.g., Llama 3) using KServe on the cluster, utilizing Scale-to-Zero to handle bursty workloads efficiently.
Deployment Reference: Create the Terraform/Manifests required to deploy this entire stack on Oracle Cloud Infrastructure (OCI), serving as a reproducible reference for the community.
Future Vision (Context for the Contributor): While beyond the immediate GSoC scope, this project lays the foundation for advanced capabilities:

Fine-Tuning & Routing: Future iterations will use KFP to fine-tune specialized "Router" models that direct queries to specific agents.
Security (MCP & Istio): We envision integrating the Model Context Protocol (MCP) and using Istio sidecars to secure agent-to-tool communication.
The GSoC contributor is building the bedrock layer that these future innovations will stand upon.

Community Value:

"Golden Data" Standard: By curating the data for this agent, we will identify gaps in our documentation and create a trusted dataset of "verified" configurations that the community can use to benchmark their own internal platforms.
Helm Alignment: This project will validate the new community Helm charts by acting as a "consumer," providing feedback on their ease of deployment in a complex GenAI stack.
Platform Alignment: We will work closely with Tarek Abouzeid to align with the Kubeflow Platform Documentation. The project must clearly separate Core Kubeflow Services (portable) from Cloud-Specific Adapters (OCI), ensuring the agentic architecture remains portable for any user.
Ideas and references:

Current Repo: kubeflow/docs-agent
Platform Standards: Kubeflow Platform Docs
Infrastructure: Terraform OCI Provider Docs
Difficulty: Hard

Size: 350 hours

Skills Required/Preferred:

Python (Backend, Agent logic)
Kubeflow (Pipelines, KServe)
GenAI/LLM Ops (RAG, Vector Databases)
Infrastructure (Terraform, Docker, Kubernetes)
Communication (Ability to document architectural decisions clearly)

--
Mentor Plan

running, deploying, and managing a REAL Milvus cluster. 
Use Helm Charts managed by ArgoCD.
Scoping & Scaling
Performance expectations and tuning

Repository Structure
agent-app
agent-infra

WebSockets? (Streaming & UX)
Token Streaming: LLMs generate text token-by-token. WebSockets push these immediately. HTTP requires polling (slow) or Server-Sent Events (SSE) (one-way).
The "Session" is the Socket: The connection itself represents the conversation state. If the socket is open, the context is alive.
RAG isn't instant. The agent needs to "Search Milvus," "Re-rank," and "Think." A long-lived pipe lets the server push status updates ("Searching knowledge base...") without client polling.
You must configure your Ingress (Nginx/ALB) to increase timeout limits (e.g., 300s+) so it doesn't kill the socket while the LLM is "thinking."
Do we need a connections DB for sessions?

Auth at Handshake: Authenticate once via the HTTP Header or Query Param during the initial protocol upgrade (Connection: Upgrade).
Trust the Pipe: Once the socket is established, do not re-authenticate every message. The open connection is the proof of identity.
e client must handle auto-reconnects. If the socket drops, the client immediately initiates a new handshake (re-submitting the token).

Identity: Session ID (UUID) is mandatory
Client generates ID -> Server uses it as the routing key.
Hot State (Redis): Mandatory.
Role: Pub/Sub message broker.
Flow: API Pod subscribes to channel:session_id. Worker publishes LLM tokens to channel:session_id.
Why: Solves K8s "split brain" where the Pod holding the socket isn't the one generating the answer. ( Could be wrong on this one need someone to delve into the depths of WS for me) 

Serving the LLM
Do we use Kserve and does it support all the above or do we need some glue?

Security:
Rate limit our docs-bot 
Guardrails for our docs bot (https://github.com/NVIDIA-NeMo/Guardrails) 
Red teaming? https://github.com/NVIDIA/garak 

Out-of-Scope
long-lived memory ( users won't be "logging into" our docs bot.