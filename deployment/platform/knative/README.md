Enable Knative in Workspace

kubectl create -n ${WORKSPACE} -f knative-override.yaml
nkp create appdeployment knative --app knative-1.17.0 -w ${WORKSPACE} --config-overrides knative-config-overrides

# Knative Eventing Kafka Broker

kubectl apply -f https://github.com/knative-sandbox/eventing-kafka-broker/releases/download/knative-v1.17.0/eventing-kafka-controller.yaml
kubectl apply -f https://github.com/knative-sandbox/eventing-kafka-broker/releases/download/knative-v1.17.0/eventing-kafka-source.yaml