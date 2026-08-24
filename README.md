# mlops-pytorch-pipeline

End-to-end MLOps pipeline: train a PyTorch image classifier, containerize it with
Docker, and deploy training + serving on Kubernetes.

## Architecture

```
                     ┌─────────────────────┐
                     │   GitHub Repo        │
                     │  (feature branches   │
                     │   → PRs → main)       │
                     └──────────┬───────────┘
                                │ CI (GitHub Actions)
                                ▼
            ┌───────────────────────────────────┐
            │        Docker Images               │
            │  mlops-train:v1   mlops-serve:v1   │
            └───────────────┬───────────────────┘
                             │
                             ▼
      ┌───────────────────────────────────────────────┐
      │            Kubernetes (namespace: ml-training)  │
      │                                                 │
      │  ConfigMap ──▶ Job (training)  ──▶ checkpoints  │
      │                                    PVC          │
      │                                       │          │
      │                                       ▼          │
      │             Deployment (2 replicas, serving) ───▶ Service (ClusterIP:80)
      │                     │  liveness/readiness /health │
      │                     ▼                              │
      │                   HPA (CPU-based autoscaling)       │
      └───────────────────────────────────────────────┘
```

## Setup

1. **Clone and create a virtualenv (local dev, optional but useful for running tests):**
   ```bash
   git clone https://github.com/<you>/mlops-pytorch-pipeline.git
   cd mlops-pytorch-pipeline
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements/train.txt
   pip install pytest
   pytest tests/ -v
   ```

2. **Build and run training locally with Docker:**
   ```bash
   docker build -f docker/Dockerfile.train -t mlops-train:v1 .
   mkdir -p data checkpoints
   docker run --rm \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/checkpoints:/app/checkpoints \
     -v $(pwd)/configs:/app/configs \
     mlops-train:v1
   ```

3. **Build and run serving locally with Docker:**
   ```bash
   docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
   docker run --rm -p 8080:8080 \
     -v $(pwd)/checkpoints:/app/checkpoints \
     mlops-serve:v1

   curl http://localhost:8080/health
   curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
   ```

4. **Deploy to Kubernetes (Minikube / kind / cloud cluster):**
   ```bash
   # if using Minikube, build images inside its docker daemon first:
   # eval $(minikube docker-env)
   # docker build -f docker/Dockerfile.train -t mlops-train:v1 .
   # docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/training-job.yaml
   kubectl wait --for=condition=complete job/mlops-training-job -n ml-training --timeout=600s

   kubectl apply -f k8s/serving-deployment.yaml
   kubectl apply -f k8s/serving-service.yaml
   kubectl apply -f k8s/hpa.yaml

   kubectl get pods -n ml-training
   kubectl describe deployment model-serving -n ml-training

   kubectl port-forward svc/model-serving 8080:80 -n ml-training
   curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
   ```

## Project structure

See assignment spec — `src/` holds model/data/train/serve code, `docker/` holds the
multi-stage Dockerfiles, `k8s/` holds the manifests, `configs/` holds hyperparameters.

## Git workflow

- `main` — protected, production-ready
- `develop` — integration branch, branched from `main`
- `feature/*` — one branch per unit of work, merged into `develop` via PR, then
  `develop` → `main` via PR at milestones
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`, ...)

## Notes on GPU training (bonus)

`k8s/training-job.yaml` has commented-out `nodeSelector`/`tolerations` and a
`nvidia.com/gpu: 1` resource limit you can enable if your cluster has GPU nodes
with the NVIDIA device plugin installed.
