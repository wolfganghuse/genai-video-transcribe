# Customize these variables
IMAGE_NAME ?= transcribe-frontend
TAG ?= 0.1.0
DOCKER_REGISTRY ?= harbor.infrastructure.cloudnative.nvdlab.net/wolfgang
CONTAINER_ENGINE ?= docker # or podman

# Build the Docker image
build:
	$(CONTAINER_ENGINE) build --platform linux/amd64 -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .

# Push the Docker image to a registry
push:
	$(CONTAINER_ENGINE) push $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)

# All-in-one command to build and push
all: build push

deploy:
	kubectl delete -f ../../deployment/deployment.yaml
	kubectl apply -f ../../deployment/deployment.yaml

run:
	streamlit run app.py --server.port=8504 --server.address=0.0.0.0

docker_run:
	$(CONTAINER_ENGINE) build -t $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG) .
	$(CONTAINER_ENGINE) run --remove --name $(IMAGE_NAME) --env-file ./.env -p 8504:8504 $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(TAG)