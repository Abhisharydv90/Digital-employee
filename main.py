import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. INITIALIZE FASTAPI & SECURITY
app = FastAPI()

# This part fixes the "CORS" error by allowing your v0.dev dashboard to talk to Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. SETUP THE BRAIN (Gemini 2.0 Flash)
# Make sure you have added GOOGLE_API_KEY to your Render Environment Variables
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    verbose=True,
    temperature=0.5,
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# 3. DEFINE THE DATA FORMAT
class ProjectRequest(BaseModel):
    prompt: str

# 4. THE MAIN ENDPOINT (The Bridge)
@app.post("/run-agency")
async def run_agency(request: ProjectRequest):
    try:
        # Define your Digital Employees
        manager = Agent(
            role='Operations Manager',
            goal='Oversee the project and ensure high quality',
            backstory='An expert CEO focused on efficiency and client satisfaction.',
            llm=llm,
            verbose=True
        )

        developer = Agent(
            role='Senior Developer',
            goal='Execute technical tasks and write clean solutions',
            backstory='A specialist who turns requirements into functional results.',
            llm=llm,
            verbose=True
        )

        # Define the Task based on what your uncle types in the dashboard
        work_task = Task(
            description=request.prompt,
            expected_output="A detailed final report or solution for the client.",
            agent=developer
        )

        # Form the Crew
        crew = Crew(
            agents=[manager, developer],
            tasks=[work_task],
            process=Process.sequential,
            verbose=True
        )

        # Execute
        result = crew.kickoff()
        
        return {"status": "Success", "output": str(result)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. HEALTH CHECK (To see if the server is awake)
@app.get("/")
async def root():
    return {"message": "Sukrit Agency Brain is ONLINE"}

if __name__ == "__main__":
    import uvicorn
    # Use the port Render provides
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
