# Terraform Variables for OCI OKE Cluster

variable "tenancy_ocid" {
  description = "OCI Tenancy OCID"
  type        = string
}

variable "user_ocid" {
  description = "OCI User OCID"
  type        = string
}

variable "compartment_ocid" {
  description = "OCI Compartment OCID"
  type        = string
}

variable "region" {
  description = "OCI Region"
  type        = string
  default    = "us-ashburn-1"
}

variable "fingerprint" {
  description = "API Key Fingerprint"
  type        = string
}

variable "private_key_path" {
  description = "Path to Private Key"
  type        = string
}

# VCN Configuration
variable "vcn_cidr" {
  description = "VCN CIDR Block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "vcn_name" {
  description = "Name for VCN"
  type        = string
  default     = "oke-vcn"
}

# Subnet Configuration
variable "subnet_cidrs" {
  description = "CIDR blocks for subnets"
  type        = map(string)
  default = {
    "public_lb"  = "10.0.10.0/24"
    "private_k8s" = "10.0.20.0/24"
  }
}

# OKE Configuration
variable "cluster_name" {
  description = "OKE Cluster Name"
  type        = string
  default     = "agentic-rag-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes Version"
  type        = string
  default     = "v1.28.2"
}

variable "pods_cidr" {
  description = "Kubernetes Pods CIDR"
  type        = string
  default     = "10.244.0.0/16"
}

variable "services_cidr" {
  description = "Kubernetes Services CIDR"
  type        = string
  default     = "10.96.0.0/16"
}

# Node Pool Configuration
variable "node_shape" {
  description = "Shape for Worker Nodes"
  type        = string
  default     = "VM.Standard.E4.Flex"
}

variable "node_count" {
  description = "Number of Worker Nodes"
  type        = number
  default     = 3
}

variable "ocpus_per_node" {
  description = "OCPUs per Node"
  type        = number
  default     = 2
}

variable "memory_per_node" {
  description = "Memory per Node in GB"
  type        = number
  default     = 32
}
