terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.1"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# ----------------------------------------------------
# 1. Image Data Sources (To Pull Images if needed)
# ----------------------------------------------------

resource "docker_image" "qdrant" {
  name         = "qdrant/qdrant:latest"
  keep_locally = true
}

resource "docker_image" "redpanda" {
  name         = "docker.redpanda.com/redpandadata/redpanda:latest"
  keep_locally = true
}

# ----------------------------------------------------
# 2. Volumes
# ----------------------------------------------------

resource "docker_volume" "qdrant_storage" {
  name = "terraform_qdrant_storage"
}

# ----------------------------------------------------
# 3. Qdrant Vector DB Container
# ----------------------------------------------------

resource "docker_container" "qdrant" {
  name  = "qdrant_terraform"
  image = docker_image.qdrant.image_id

  ports {
    internal = 6333
    external = 6333
  }
  ports {
    internal = 6334
    external = 6334
  }

  volumes {
    volume_name    = docker_volume.qdrant_storage.name
    container_path = "/qdrant/storage"
  }

  restart = "unless-stopped"
}

# ----------------------------------------------------
# 4. Redpanda Streaming Platform (Kafka Compatible)
# ----------------------------------------------------

resource "docker_container" "redpanda" {
  name  = "redpanda_terraform"
  image = docker_image.redpanda.image_id

  command = [
    "redpanda", "start", "--smp", "1",
    "--reserve-memory", "0M", "--overprovisioned",
    "--node-id", "0",
    "--kafka-addr", "PLAINTEXT://0.0.0.0:49092,OUTSIDE://0.0.0.0:39092",
    "--advertise-kafka-addr", "PLAINTEXT://redpanda_terraform:49092,OUTSIDE://localhost:39092",
    "--pandaproxy-addr", "PLAINTEXT://0.0.0.0:48082,OUTSIDE://0.0.0.0:38082",
    "--advertise-pandaproxy-addr", "PLAINTEXT://redpanda_terraform:48082,OUTSIDE://localhost:38082"
  ]

  ports {
    internal = 8081
    external = 8088
  }
  ports {
    internal = 8082
    external = 8089
  }
  ports {
    internal = 39092
    external = 39092
  }
  ports {
    internal = 38082
    external = 38082
  }
  ports {
    internal = 49092
    external = 49092
  }

  restart = "unless-stopped"
}
