# Terraform Outputs

output "vcn_id" {
  description = "OCID of the created VCN"
  value       = oci_core_vcn.oke_vcn.id
}

output "vcn_cidr" {
  description = "CIDR block of the VCN"
  value       = oci_core_vcn.oke_vcn.cidr_block
}

output "public_subnet_id" {
  description = "OCID of public subnet for Load Balancers"
  value       = oci_core_subnet.public_lb_subnet.id
}

output "private_subnet_id" {
  description = "OCID of private subnet for Kubernetes nodes"
  value       = oci_core_subnet.private_k8s_subnet.id
}

output "cluster_id" {
  description = "OCID of the OKE Cluster"
  value       = oci_containerengine_cluster.oke_cluster.id
}

output "cluster_name" {
  description = "Name of the OKE Cluster"
  value       = oci_containerengine_cluster.oke_cluster.name
}

output "cluster_endpoint" {
  description = "Kubernetes API Server Endpoint"
  value       = try(oci_containerengine_cluster.oke_cluster.endpoints[0].endpoint, "")
}

output "node_pool_id" {
  description = "OCID of the Node Pool"
  value       = oci_containerengine_node_pool.oke_node_pool.id
}

output "node_count" {
  description = "Number of worker nodes"
  value       = var.node_count
}

output "kubeconfig_command" {
  description = "Command to get kubeconfig"
  value       = "oci ce cluster generate-token --cluster-id ${oci_containerengine_cluster.oke_cluster.id} --raw-filename > kubeconfig"
  sensitive   = false
}
