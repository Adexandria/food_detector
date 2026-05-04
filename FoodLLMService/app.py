from fastapi import FastAPI
from server.api import router as api_router
import uvicorn

app = FastAPI(
    title="Food LLM API",
    description="API for managing diaglogue with the user by gnerating responses using the LLM.",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Food LLM API. Use /main to start the conversation and /set-food to set the food preferences."}
    
    
app.include_router(api_router)



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)