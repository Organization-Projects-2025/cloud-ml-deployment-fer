# FER-DDAMFN Video Emotion API (FastAPI + Docker + Kubernetes)

A FastAPI service that runs facial emotion recognition on uploaded videos and returns emotion predictions.
Deployed with Kubernetes using a Deployment, a Service, health checks (liveness/readiness), and HPA (CPU-based autoscaling).

---

## API Endpoints

### Health
GET /healthz

Example:
curl http://localhost:8000/healthz


### Predict from video
POST /predict-video
Content-Type: multipart/form-data
Form field: file

Example:
curl -X POST "http://localhost:8000/predict-video" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@sample.mp4;type=video/mp4"

---

## Run locally (Docker)

### Pull from Docker Hub
docker pull <DOCKERHUB_USERNAME>/<IMAGE_NAME>:<TAG>

Example:
docker pull youssefahmed64/fer-ddamfn:cpu

### Run
docker run --rm -p 8000:8000 --name fer-ddamfn youssefahmed64/fer-ddamfn:cpu

Open:
- http://localhost:8000/healthz
- http://localhost:8000/docs

---

## Kubernetes deployment (kubectl)

### 1) Create Deployment
kubectl create deployment fer-ddamfn --image=youssefahmed64/fer-ddamfn:cpu

### 2) Create Service
(Use NodePort for easy browser access)
kubectl expose deployment fer-ddamfn --name=fer-ddamfn-svc --type=NodePort --port=8000 --target-port=8000

### 3) Set CPU/Memory requests & limits (required for HPA CPU scaling)
kubectl set resources deployment/fer-ddamfn --requests=cpu=200m,memory=256Mi --limits=cpu=500m,memory=512Mi

### 4) Health checks (Liveness + Readiness)
This project uses /healthz for both probes.
If your kubectl version does not support "kubectl set probe", apply probes via "kubectl patch" (patch files tracked in repo).

Liveness:
kubectl patch deployment/fer-ddamfn --type=strategic --patch-file ./k8s/liveness-patch.json

Readiness:
kubectl patch deployment/fer-ddamfn --type=strategic --patch-file ./k8s/readiness-patch.json

Verify:
kubectl describe pod -l app=fer-ddamfn

### 5) HPA (min=1, max=5, CPU target=50%)
kubectl autoscale deployment fer-ddamfn --min=1 --max=5 --cpu-percent=50

Verify:
kubectl get hpa
kubectl top pods

---

## Export Kubernetes YAMLs (generated from cluster)
After creating resources, export the live manifests:

kubectl get deploy fer-ddamfn -o yaml > k8s/deployment.yaml
kubectl get svc fer-ddamfn-svc -o yaml > k8s/service.yaml
kubectl get hpa fer-ddamfn -o yaml > k8s/hpa.yaml

---

## Notes (public repo)
- Do NOT commit secrets (.env, tokens, keys).
- Large model weights can be excluded using .gitignore if needed.
- Ensure the image tag in Kubernetes matches the one pushed to Docker Hub.

---
