#!/usr/bin/env bash
# Create a local kind cluster and deploy llmserve with mock workers, KEDA autoscaling and the
# KubeRay operator (used to validate the GPU RayCluster manifest; no Ray pods are started).
#
#   ./scripts/kind_up.sh            # create + deploy
#   kind delete cluster --name llmserve   # tear down
set -euo pipefail
cd "$(dirname "$0")/.."
KIND=${KIND:-kind}
HELM=${HELM:-helm}
CLUSTER=llmserve

if ! $KIND get clusters 2> /dev/null | grep -qx "$CLUSTER"; then
  $KIND create cluster --name "$CLUSTER" --wait 120s
fi
kubectl config use-context "kind-$CLUSTER" > /dev/null

docker build -q -t llmserve:dev . > /dev/null
$KIND load docker-image llmserve:dev --name "$CLUSTER"

$HELM repo add kedacore https://kedacore.github.io/charts > /dev/null 2>&1 || true
$HELM repo add kuberay https://ray-project.github.io/kuberay-helm/ > /dev/null 2>&1 || true
$HELM repo update > /dev/null
$HELM upgrade --install keda kedacore/keda --version 2.16.1 -n keda --create-namespace --wait \
  --set resources.operator.requests.memory=64Mi --set resources.metricServer.requests.memory=64Mi
$HELM upgrade --install kuberay-operator kuberay/kuberay-operator --version 1.3.2 -n kuberay --create-namespace --wait

kubectl apply -k deploy/k8s/overlays/kind
kubectl -n llmserve rollout status deploy/llmserve-worker --timeout=120s
kubectl -n llmserve rollout status deploy/llmserve-router --timeout=120s

# Validate the GPU RayCluster manifest against the real CRD without creating it.
kubectl apply --dry-run=server -f deploy/kuberay/raycluster-vllm.yaml

kubectl -n llmserve get pods,svc,scaledobject
echo "Port-forward the router with: kubectl -n llmserve port-forward svc/llmserve-router 8000:8000"
