# AI-HR-Agent
An intelligent multi-agent system for automated hiring workflows, powered by LangGraph, advanced NLP, and Tavily Search integration.

## 🌟 Overview

The Enhanced AI HR Agent is a sophisticated hiring automation system that leverages state-of-the-art AI technologies to streamline the recruitment process. It uses a multi-agent architecture with intelligent intent classification to provide both conversational assistance and comprehensive hiring workflow automation.

### Key Highlights
- **Multi-Agent Orchestration** with specialized agents for different tasks
- **Intent Classification** to intelligently route queries
- **Conditional Tool Usage** for optimal performance
- **Advanced NLP** with spaCy for role detection
- **Market Research** via Tavily Search API
- **Session Memory** for conversation continuity
- **Professional Export** capabilities (JSON/Markdown)

## 🏗️ Architecture

### System Architecture Overview

```
┌─────────────────────────────────────┐
│     Presentation Layer (Gradio)     │
├─────────────────────────────────────┤
│    Orchestration Layer (LangGraph)  │
├─────────────────────────────────────┤
│      Agent Layer (Multi-Agent)      │
├─────────────────────────────────────┤
│        Tool Layer (3 Tools)         │
├─────────────────────────────────────┤
│    External Services (APIs)         │
└─────────────────────────────────────┘
```

### Multi-Agent System Architecture

The system employs a **Multi-Agent System (MAS)** architecture with the following specialized agents:

1. **Intent Classifier** 🧠
   - Classifies user input as conversational or hiring-related
   - Routes queries to appropriate processing paths
   - Prevents unnecessary tool calls for simple queries

2. **Coordinator Agent** 🎯
   - Orchestrates the overall workflow
   - Manages session initialization
   - Coordinates between different agents

3. **NLP Analyzer Agent** 🔍
   - Performs advanced role detection using NLP
   - Extracts requirements from natural language
   - Generates confidence scores for detected roles

4. **Research Agent** 📊
   - Conducts market research via Tavily Search
   - Aggregates salary data and market insights
   - Provides competitive intelligence

5. **Content Agent** 📝
   - Generates job descriptions
   - Creates recruitment email templates
   - Produces role-specific content

6. **Integration Agent** ⚙️
   - Creates adaptive hiring workflows
   - Builds timeline-based checklists
   - Handles workflow complexity analysis

7. **Memory Manager Agent** 💾
   - Maintains conversation continuity
   - Tracks session state
   - Enables follow-up conversations

8. **Output Formatter Agent** 📋
   - Structures final output
   - Handles export formatting
   - Prepares professional reports

### LangGraph Workflow Architecture

The system uses a **Conditional Directed Acyclic Graph (DAG)** workflow:

```
START 
  └─> Intent Classifier
        ├─> [conversation] → Conversational Agent → Output Formatter → END
        └─> [hiring] → Coordinator → Memory Manager → NLP Analyzer
                                                         ├─> [clarification] → Requirements Check → END
                                                         └─> [proceed] → Research → Content → Integration → Synthesis → END
```

### Tool Architecture

Three specialized tools provide core functionality:

1. **TavilySearchTool** 🔍
   - Intelligent market research
   - Real-time salary data
   - Industry insights and trends

2. **EnhancedEmailWriterTool** 📧
   - Role-specific email templates
   - AI-powered content generation
   - Multiple email types (outreach, interview, offer)

3. **EnhancedChecklistBuilderTool** 📋
   - Adaptive workflow generation
   - Timeline optimization
   - Phase-based hiring plans

## 🚀 Features

### Core Features
- ✅ **Intelligent Role Detection** - Extracts roles from natural language with confidence scoring
- ✅ **Clarifying Questions** - Asks for missing information when needed
- ✅ **Job Description Generation** - Creates tailored job descriptions for each role
- ✅ **Market Research** - Provides salary ranges and market insights
- ✅ **Hiring Workflow** - Generates phase-based hiring checklists with timelines
- ✅ **Email Templates** - Creates recruitment emails for different stages
- ✅ **Session Memory** - Maintains context across conversations
- ✅ **Export Capabilities** - JSON and Markdown report generation

### Enhanced Features
- 🎯 **Intent Classification** - Distinguishes between conversational and hiring queries
- ⚡ **Conditional Processing** - Optimizes performance by avoiding unnecessary tool calls
- 🧠 **Advanced NLP** - Uses spaCy for sophisticated text analysis
- 📊 **Adaptive Workflows** - Adjusts complexity based on role requirements
- 🔄 **Follow-up Support** - Handles incremental updates and changes
- 📈 **Analytics** - Tracks session performance and success metrics

## 📋 Requirements

### System Requirements
- Python 3.11+
- 4GB RAM minimum (8GB recommended for optimal performance)
- Internet connection for API calls

### API Keys Required
- **OpenAI API Key** - For LLM capabilities
- **Tavily API Key** - For market research (optional, will use mock data if not provided)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/enhanced-hr-agent.git
cd enhanced-hr-agent
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

5. **Set up environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "TAVILY_API_KEY=your_tavily_api_key_here" >> .env
```

## 🎮 Usage

### Running the Application

```python
# In Jupyter/Colab
%run enhanced_hr_agent.py

# Or directly
python enhanced_hr_agent.py
```

### Using the Gradio Interface

1. **Launch the interface**:
```python
launch_enhanced_hr_agent()
```

2. **Access the interface**:
   - Local: `http://localhost:7860`
   - Public URL will be displayed if running in Colab

### Example Interactions

#### Conversational Query
```
User: "Hi, how can you help?"
System: Provides overview of capabilities without calling any tools
```

#### Hiring Request
```
User: "I need to hire a founding engineer and a GenAI intern for my AI startup"
System: 
- Detects roles with confidence scores
- Conducts market research
- Generates job descriptions
- Creates hiring workflow
- Provides email templates
```

### Programmatic Usage

```python
# Initialize orchestrator
orchestrator = EnhancedHRAgentOrchestrator()

# Process hiring request
result = await orchestrator.process_hiring_request(
    user_input="I need a senior ML engineer",
    company_context={
        "company_name": "AI Startup Inc",
        "industry": "AI/ML",
        "stage": "Series A",
        "team_size": 20
    }
)

# Access results
roles = result.get("detected_roles", [])
market_data = result["agent_outputs"]["research"]["research_data"]
workflow = result["agent_outputs"]["integration"]["hiring_checklist"]
```

## 📊 Export Formats

### JSON Export
Structured data export containing:
- Session metadata
- Detected roles with confidence scores
- Market research data
- Generated content
- Workflow details
- Analytics

### Markdown Export
Professional report format with:
- Executive summary
- Role analysis
- Market insights
- Hiring timeline
- Next steps

## 🔧 Configuration

### Role Detection Configuration
The system recognizes various role patterns including:
- Engineering roles (Founding Engineer, Senior Engineer, Full-Stack)
- AI/ML roles (AI Engineer, ML Engineer, GenAI Engineer, Data Scientist)
- Product roles (Product Manager)
- Design roles (UX Designer)

### Workflow Complexity Levels
- **Standard roles**: 30-day baseline timeline
- **Senior roles**: +5 days for additional vetting
- **AI/ML roles**: +7 days for technical assessment
- **Founding roles**: +14 days for comprehensive evaluation

## 🐛 Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
```bash
python -m spacy download en_core_web_sm
```

2. **API Key Errors**
- Ensure API keys are properly set in environment variables
- System will use mock data if Tavily key is missing

3. **Memory Issues**
- Reduce batch processing size
- Clear session cache periodically

4. **Export Not Working**
- Ensure session is complete before exporting
- Check that session_id is valid

## 📈 Performance Optimization

- **Caching**: Results are cached at multiple levels
- **Conditional Processing**: Unnecessary agents are skipped
- **Async Operations**: All API calls are asynchronous
- **Memory Management**: Sessions are stored efficiently

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
## 🙏 Acknowledgments

- **LangChain** & **LangGraph** for the agent framework
- **OpenAI** for LLM capabilities
- **Tavily** for intelligent search
- **spaCy** for NLP processing
- **Gradio** for the web interface
