#!/bin/bash

cd "$(dirname "$0")"
uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!

cd frontend
node server.js &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
  echo "Stopping servers..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  exit
}

# Register the cleanup function for program termination
trap cleanup INT TERM

echo "Frontend with PID $FRONTEND_PID: http://localhost:3000"
echo "Backend with PID $BACKEND_PID: http://localhost:8000"
printf "Check backend.log for backend server output.\nPress Ctrl+C to stop both servers.\n"

wait