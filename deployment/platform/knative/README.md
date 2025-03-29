# Enable Knative in Workspace
## Run in Management Cluster

### Deploy istio
nkp create appdeployment istio --app istio-1.23.3 -w wl01

### Deploy knative serving/eventing
kubectl create -n ${WORKSPACE} -f knative-override.yaml
nkp create appdeployment knative --app knative-1.17.0 -w ${WORKSPACE} --config-overrides knative-config-overrides

# Knative Eventing Kafka Broker
## Run in Managed Cluster
kubectl apply -f https://github.com/knative-sandbox/eventing-kafka-broker/releases/download/knative-v1.17.0/eventing-kafka-controller.yaml
kubectl apply -f https://github.com/knative-sandbox/eventing-kafka-broker/releases/download/knative-v1.17.0/eventing-kafka-source.yaml