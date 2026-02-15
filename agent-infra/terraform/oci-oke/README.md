# OCI OKE Terraform

Terraform configuration to provision Oracle Kubernetes Engine (OKE) cluster on Oracle Cloud Infrastructure (OCI).

## Overview

This Terraform creates:
- Virtual Cloud Network (VCN) with subnets
- Internet Gateway and NAT Gateway
- Security Lists for networking
- OKE Cluster with managed control plane
- Node Pool with configurable worker nodes

## Prerequisites

1. **OCI Account** - Sign up at [oracle.com/cloud](https://www.oracle.com/cloud/)
2. **Terraform** - Install via `brew install terraform` or [download](https://www.terraform.io/downloads)
3. **OCI CLI** - Install via `brew install oci-cli` (optional, for generating kubeconfig)

## Setup

### 1. Get OCI Credentials

1. Login to OCI Console
2. Go to Identity → Users → Your User
3. API Keys → Add API Key
4. Download the private key
5. Copy the Tenancy OCID, User OCID, and Fingerprint

### 2. Configure Variables

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` and fill in your credentials.

### 3. Initialize Terraform

```bash
cd agent-infra/terraform/oci-oke
terraform init
```

### 4. Plan and Apply

```bash
# See what will be created
terraform plan

# Create the infrastructure
terraform apply
```

## Usage

### Get kubeconfig

```bash
oci ce cluster generate-token --cluster-id <cluster_ocid> --raw-filename > kubeconfig
export KUBECONFIG=kubeconfig

# Verify connection
kubectl get nodes
```

### Scale Node Pool

Edit `variable.tf` and change `node_count`, then run:

```bash
terraform apply
```

### Destroy Infrastructure

```bash
terraform destroy
```

## Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `tenancy_ocid` | OCI Tenancy OCID | Required |
| `user_ocid` | OCI User OCID | Required |
| `compartment_ocid` | OCI Compartment OCID | Required |
| `region` | OCI Region | `us-ashburn-1` |
| `vcn_cidr` | VCN CIDR Block | `10.0.0.0/16` |
| `cluster_name` | OKE Cluster Name | `agentic-rag-cluster` |
| `kubernetes_version` | Kubernetes Version | `v1.28.2` |
| `node_shape` | Worker Node Shape | `VM.Standard.E4.Flex` |
| `node_count` | Number of Worker Nodes | `3` |
| `ocpus_per_node` | OCPUs per Node | `2` |
| `memory_per_node` | Memory per Node (GB) | `32` |

## Outputs

After `terraform apply`, you can use:

- `cluster_endpoint` - Kubernetes API server URL
- `kubeconfig_command` - Command to generate kubeconfig
- `vcn_id` - VCN OCID
- `cluster_id` - Cluster OCID

## Costs

This configuration creates paid OCI resources:
- OKE Cluster (free control plane)
- Compute instances for nodes
- Load balancers
- NAT Gateway

Use [OCI Pricing Calculator](https://www.oracle.com/cloud/price-calculator/) to estimate costs.
