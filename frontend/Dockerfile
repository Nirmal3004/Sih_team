# Development build for the DeepWeather AI dashboard.
# Vite serves the app on port 5173 and talks to the FastAPI backend on 8000.
FROM node:22-alpine

WORKDIR /app

# Install dependencies first so the layer is cached across code changes.
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
