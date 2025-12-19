# FER-DDAMFN Video Emotion Recognition (FastAPI) — Docker + Kubernetes

This repository contains a FastAPI application that runs a DDAMFN-based facial emotion recognition model and exposes:

- Health endpoint: GET /healthz
- Video inference endpoint: POST /predict-video (multipart/form-data)

It is containerized with Docker and deployed to Kubernetes using Deployment + Service, with:

- Liveness + Readiness probes (HTTP GET /healthz on port 8000)
- Horizontal Pod Autoscaler (HPA): min 1, max 5, target CPU 50%

---

## 1) Requirements

- Docker Desktop (or Docker Engine)
- Kubernetes cluster (Docker Desktop Kubernetes / Minikube / etc.)
- kubectl

---

## 2) API Endpoints

### Health

- GET /healthz
  Example:
  curl http://localhost:8000/healthz

### Predict from Video

- POST /predict-video
- Body: multipart/form-data
- Field name: file
  Example:
  curl -X POST "http://localhost:8000/predict-video" \
   -H "accept: application/json" \
   -H "Content-Type: multipart/form-data" \
   -F "file=@sample.mp4;type=video/mp4"

---

## 3) Run locally with Docker

### Pull image from Docker Hub

docker pull youssefahmed64/fer-ddamfn:cpu

### Run container (port 8000)

docker run --rm -p 8000:8000 --name fer-ddamfn youssefahmed64/fer-ddamfn:cpu

### Test

curl http://localhost:8000/healthz

Open Swagger UI:
http://localhost:8000/docs

---

## 4) Dockerization steps (build + push)

### Build

cd C:\Games\cloud-project
docker build -t youssefahmed64/fer-ddamfn:cpu .

### Run locally

docker run --rm -p 8000:8000 youssefahmed64/fer-ddamfn:cpu

### Login (if needed)

docker login

### Push

docker push youssefahmed64/fer-ddamfn:cpu

(Optional) Inspect layers
docker image history --no-trunc youssefahmed64/fer-ddamfn:cpu

---

## 5) Kubernetes deployment steps

### Check cluster

kubectl cluster-info
kubectl get nodes

### Create Deployment

kubectl create deployment fer-ddamfn --image=youssefahmed64/fer-ddamfn:cpu

### Expose Service (NodePort example)

kubectl expose deployment fer-ddamfn --name=fer-ddamfn-svc --type=NodePort --port=8000 --target-port=8000

### Set resources (required for CPU-based HPA)

kubectl set resources deployment/fer-ddamfn --requests=cpu=200m,memory=256Mi --limits=cpu=500m,memory=512Mi
kubectl rollout status deployment/fer-ddamfn

---

## 6) Health checks (Liveness + Readiness)

This app exposes /healthz, so probes must hit /healthz on port 8000.

### Create liveness patch file and apply (PowerShell)

cd C:\Games\cloud-project\k8s

$liv = '{"spec":{"template":{"spec":{"containers":[{"name":"fer-ddamfn","livenessProbe":{"httpGet":{"path":"/healthz","port":8000},"initialDelaySeconds":10,"periodSeconds":10,"timeoutSeconds":2,"failureThreshold":3}}]}}}}'
[System.IO.File]::WriteAllText("$PWD\liveness-patch.json", $liv, (New-Object System.Text.UTF8Encoding($false)))
kubectl patch deployment/fer-ddamfn --type=strategic --patch-file .\liveness-patch.json

### Create readiness patch file and apply (PowerShell)

$red = '{"spec":{"template":{"spec":{"containers":[{"name":"fer-ddamfn","readinessProbe":{"httpGet":{"path":"/healthz","port":8000},"initialDelaySeconds":5,"periodSeconds":5,"timeoutSeconds":2,"failureThreshold":3}}]}}}}'
[System.IO.File]::WriteAllText("$PWD\readiness-patch.json", $red, (New-Object System.Text.UTF8Encoding($false)))
kubectl patch deployment/fer-ddamfn --type=strategic --patch-file .\readiness-patch.json

### Verify probes

kubectl describe pod -l app=fer-ddamfn | findstr /i "Liveness Readiness"

---

## 7) Horizontal Pod Autoscaler (HPA)

Create HPA (min 1, max 5, target CPU 50%):
kubectl autoscale deployment fer-ddamfn --cpu-percent=50 --min=1 --max=5

Verify:
kubectl get hpa
kubectl top pods

---

## 8) Export YAMLs (generated from the live cluster)

cd C:\Games\cloud-project\k8s
kubectl get deploy fer-ddamfn -o yaml > deployment.yaml
kubectl get svc fer-ddamfn-svc -o yaml > service.yaml
kubectl get hpa fer-ddamfn -o yaml > hpa.yaml

---

## 9) How to verify you satisfied the requirements

### Deployment + Service exist

kubectl get deploy fer-ddamfn
kubectl get svc fer-ddamfn-svc
kubectl get pods -l app=fer-ddamfn

### Probes are applied

kubectl describe pod -l app=fer-ddamfn | findstr /i "Liveness Readiness"

### HPA is configured

kubectl get hpa fer-ddamfn
kubectl describe hpa fer-ddamfn
kubectl top pods

---

## 10) Troubleshooting

### Swagger UI: “Failed to fetch” (CORS / Network Failure)

- Ensure you open Swagger from the running API: http://localhost:8000/docs
- Confirm OpenAPI schema loads: http://localhost:8000/openapi.json
- If calling from a different frontend origin/port, configure CORS in FastAPI (CORSMiddleware).

### Rollout stuck at 1/2 pods

Usually probe path mismatch (404). Ensure probes hit /healthz, not /health or /ready:
kubectl describe pod <pod-name>

---

## Suggested repo name

fer-ddamfn-k8s-deployment
