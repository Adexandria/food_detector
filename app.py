from fastapi import FastAPI
from server.api import router as api_router
import uvicorn

app = FastAPI(
    title="Food Classification API",
    description="API for classifying the food.",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Classification API. Use /api/detect to classify food images."}
    
app.include_router(api_router)



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)