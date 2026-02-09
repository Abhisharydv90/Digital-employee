import os
from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware

# 1. SETUP THE BRAIN (LLM)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# 2. THE API BRIDGE
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ProjectRequest(BaseModel):
    prompt: str

@app.post("/run-agency")
async def start_agency(request: ProjectRequest):
    # YOUR DIGITAL EMPLOYEES
    manager = Agent(role='Manager', goal='Oversee project', backstory='CEO', llm=llm)
    developer = Agent(role='Developer', goal='Write clean code', backstory='Expert', llm=llm)
    
    t1 = Task(description=request.prompt, expected_output="Final solution", agent=developer)
    
    # Starting the Crew
    crew = Crew(agents=[manager, developer], tasks=[t1], process=Process.hierarchical, manager_llm=llm)
    result = crew.kickoff()
    
    return {"status": "Success", "output": str(result)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
