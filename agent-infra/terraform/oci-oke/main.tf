# OCI Provider Configuration
terraform {
  required_version = ">= 1.0.0"

  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 5.0"
    }
  }
}

provider "oci" {
  tenancy_ocid   = var.tenancy_ocid
  user_ocid      = var.user_ocid
  compartment_ocid = var.compartment_ocid
  region         = var.region
  fingerprint    = var.fingerprint
  private_key_path = var.private_key_path
}

# VCN Creation
resource "oci_core_vcn" "oke_vcn" {
  cidr_block     = var.vcn_cidr
  compartment_id = var.compartment_ocid
  display_name   = var.vcn_name
  dns_label      = "okevcn"
}

# Internet Gateway
resource "oci_core_internet_gateway" "oke_igw" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.oke_vcn.id
  display_name   = "${var.vcn_name}-igw"
  enabled        = true
}

# NAT Gateway
resource "oci_core_nat_gateway" "oke_nat" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.oke_vcn.id
  display_name   = "${var.vcn_name}-nat"
}

# Public Subnet for Load Balancers
resource "oci_core_subnet" "public_lb_subnet" {
  cidr_block                = var.subnet_cidrs["public_lb"]
  compartment_id            = var.compartment_ocid
  vcn_id                    = oci_core_vcn.oke_vcn.id
  display_name              = "public-lb-subnet"
  dns_label                 = "publblb"
  prohibit_public_ip_on_vnic = false
  route_table_id            = oci_core_route_table.public_rt.id
  security_list_ids         = [oci_core_security_list.public_lb_sl.id]
}

# Private Subnet for Kubernetes Nodes
resource "oci_core_subnet" "private_k8s_subnet" {
  cidr_block                = var.subnet_cidrs["private_k8s"]
  compartment_id            = var.compartment_ocid
  vcn_id                    = oci_core_vcn.oke_vcn.id
  display_name              = "private-k8s-subnet"
  dns_label                 = "privatek8s"
  prohibit_public_ip_on_vnic = true
  route_table_id            = oci_core_route_table.private_rt.id
  security_list_ids         = [oci_core_security_list.private_k8s_sl.id]
}

# Public Route Table
resource "oci_core_route_table" "public_rt" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.oke_vcn.id
  display_name   = "public-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_internet_gateway.oke_igw.id
  }
}

# Private Route Table (via NAT for outbound)
resource "oci_core_route_table" "private_rt" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.oke_vcn.id
  display_name   = "private-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    destination_type  = "CIDR_BLOCK"
    network_entity_id = oci_core_nat_gateway.oke_nat.id
  }
}

# Security List for Public LB Subnet
resource "oci_core_security_list" "public_lb_sl" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.oke_vcn.id
  display_name   = "public-lb-sl"

  egress_security_rules {
    destination      = "0.0.0.0/0"
    destination_type = "CIDR_BLOCK"
  }

  ingress_security_rules {
    protocol  = "6"  # TCP
    source    = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }

  ingress_security_rules {
    protocol  = "6"  # TCP
    source    = "0.0.0.0/0"
    tcp_options {
      min = 443
      max = 443
    }
  }
}

# Security List for Private K8s Subnet
resource "oci_core_security_list" "private_k8s_sl" {
  compartment_id = var.compartment_ocid
  vcn_id         = oci_core_vcn.oke_vcn.id
  display_name   = "private-k8s-sl"

  egress_security_rules {
    destination      = "0.0.0.0/0"
    destination_type = "CIDR_BLOCK"
  }

  # Allow SSH from bastion or VPN
  ingress_security_rules {
    protocol  = "6"  # TCP
    source    = var.vcn_cidr
    tcp_options {
      min = 22
      max = 22
    }
  }

  # Allow all traffic within VCN
  ingress_security_rules {
    protocol  = "all"
    source    = var.vcn_cidr
  }
}
