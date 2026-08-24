# End-to-End Validation

## Local Docker
- Training image built and ran successfully, produced classifier_v1.pt checkpoint
- Serving image built and ran successfully

## Kubernetes (Minikube)
- namespace, configmap, training-job, serving-deployment, serving-service, hpa all applied successfully
- Training Job completed: `mlops-training-job-wtngn   0/1   Completed`
- Serving Deployment: 2/2 replicas Running and Ready

## API validation
Health check:
    curl http://localhost:8080/health
    {"status":"ok"}

Prediction:
    curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
    {"predicted_class":"cat","probabilities":{"airplane":0.1667,"automobile":0.0305,
    "bird":0.0483,"cat":0.3242,"deer":0.0383,"dog":0.1278,"frog":0.0584,
    "horse":0.0146,"ship":0.1314,"truck":0.0597}}
