# Customize these variables
IMAGE_NAME ?= transcribe-backend
TAG ?= 0.1.0
DOCKER_REGISTRY ?= harbor.infrastructure.cloudnative.nvdlab.net/wolfgang
CONTAINER_ENGINE ?= docker # or podman

# Build the Docker image
build:
	$(CONTAINER_ENGINE) build --platform linux/amd64 -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

# Push the Docker image to a registry
push:
	$(CONTAINER_ENGINE) push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)


deploy:
	kubectl delete -f ../../deployment/service.yaml
	kubectl apply -f ../../deployment/service.yaml

# All-in-one command to build and push
all: build push deploy

run:
	$(CONTAINER_ENGINE) run -it -p 8080:8080 $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)
