#!/bin/bash

set -e

IMAGE_URI="europe-west1-docker.pkg.dev/avisia-certification-ml-yde/chicago-taxis-demo/forecasting-pipeline:latest"

echo "🔍 Vérification du contexte Docker..."
if ! docker info >/dev/null 2>&1; then
  echo "⚙️  Docker daemon inactif. Tentative de démarrage via Colima..."
  colima start
  docker context use colima
fi

echo "🚧 Construction de l’image : $IMAGE_URI"
docker build --no-cache -t ${IMAGE_URI} .

echo "📤 Push de l’image vers Artifact Registry"
docker push ${IMAGE_URI}

echo "✅ Terminé avec succès !"
