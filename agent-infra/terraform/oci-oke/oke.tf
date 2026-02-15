variable "node_image_id" {
  description = "OCID of the image for worker nodes. Leave empty to use OKE default."
  type        = string
  default     = ""
}

# OKE Cluster Definition
resource "oci_containerengine_cluster" "oke_cluster" {
  compartment_id     = var.compartment_ocid
  name               = var.cluster_name
  kubernetes_version = var.kubernetes_version

  options {
    service_lb_subnet_ids = [oci_core_subnet.public_lb_subnet.id]

    kubernetes_network_config {
      pods_cidr     = var.pods_cidr
      services_cidr = var.services_cidr
    }

    # Enable VCN-native pod networking (more secure)
    is_pod_security_policy_enabled = false
  }

  vcn_id = oci_core_vcn.oke_vcn.id

  # Wait for cluster to be provisioned
  timeouts {
    create = "30m"
    update = "30m"
    delete = "30m"
  }
}

# OKE Node Pool
resource "oci_containerengine_node_pool" "oke_node_pool" {
  compartment_id = var.compartment_ocid
  cluster_id     = oci_containerengine_cluster.oke_cluster.id
  name           = "${var.cluster_name}-nodepool"

  kubernetes_version = var.kubernetes_version

  node_shape = var.node_shape

  # Flexible instance configuration
  node_config_details {
    size                     = var.node_count
    subnet_ids               = [oci_core_subnet.private_k8s_subnet.id]

    # Flexible shape configuration
    flexible_ocpu_settings {
      ocpus = var.ocpus_per_node
    }

    memory_in_gbs = var.memory_per_node

    # Node pool placement
    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.private_k8s_subnet.id
    }

    # CNI options
    node_pool_pod_network_options {
      cni_type = "FLANNEL_OVERLAY"
    }
  }

  # Node source - use OCI Image
  node_source_details {
    image_id                = var.node_image_id != "" ? var.node_image_id : null
    source_type             = "IMAGE"
    node_volume_size_in_gbs = 100
  }

  timeouts {
    create = "30m"
    update = "30m"
    delete = "30m"
  }
}

# Get available availability domains
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# Optional: Get latest OKE node image options
data "oci_containerengine_node_pool_option" "oke_node_pool_options" {
  cluster_id = oci_containerengine_cluster.oke_cluster.id

  node_pool_option_id = "oke-node-pool-options"
}
