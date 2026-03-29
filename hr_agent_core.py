# ==================== HR Agent Core Module ====================
# Extracted from AI_HR_agent notebook - importable module for Streamlit

# Import enhanced libraries
import os
import json
import asyncio
import time
import nest_asyncio
import spacy
import re
from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

# Load environment variables from .env file
load_dotenv()

# Fix event loop for notebook/Streamlit compatibility
nest_asyncio.apply()

# Load spaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
    print("SpaCy model loaded for advanced NLP")
except OSError:
    print("SpaCy model not found, using basic extraction")
    nlp = None

# ==================== CELL SEPARATOR ====================

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    NLP_ANALYZER = "nlp_analyzer"  # New: Advanced role detection
    REQUIREMENTS = "requirements"
    RESEARCH = "research"
    CONTENT = "content"
    INTEGRATION = "integration"
    MEMORY_MANAGER = "memory_manager"  # New: Session management
    OUTPUT_FORMATTER = "output_formatter"  # New: Export management

@dataclass
class EnhancedRoleRequirement:
    role_name: str
    role_category: str  # 'ai_ml', 'engineering', 'product', 'design', etc.
    priority: int = 1
    confidence_score: float = 0.8  # NLP confidence

    # Enhanced attributes
    technical_level: str = ""  # junior, mid, senior, principal, staff
    specialization: List[str] = field(default_factory=list)
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    timeline_days: Optional[int] = None
    skills_required: List[str] = field(default_factory=list)
    skills_nice_to_have: List[str] = field(default_factory=list)

    # Context attributes
    location_preference: str = ""
    remote_flexibility: float = 1.0
    equity_expectation: Optional[str] = None
    urgency_score: float = 0.5
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ConversationMemory:
    session_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    company_context: Dict[str, Any] = field(default_factory=dict)
    previous_outputs: List[Dict[str, Any]] = field(default_factory=list)
    change_requests: List[str] = field(default_factory=list)

class EnhancedHRAgentState(TypedDict):
    # Core conversation with memory
    messages: Annotated[list, add_messages]
    session_id: str
    conversation_memory: Dict[str, Any]

    # Enhanced role analysis
    detected_roles: List[Dict[str, Any]]
    role_analysis_confidence: float
    clarifications_needed: List[str]

    # Company context
    company_context: Dict[str, Any]
    industry_insights: Dict[str, Any]

    # Multi-agent coordination
    current_agent: str
    agent_outputs: Dict[str, Any]
    workflow_stage: str
    completion_percentage: float

    # Session management
    is_followup: bool
    change_detection: Dict[str, Any]
    incremental_updates: List[str]

    # Output management
    structured_output: Dict[str, Any]
    export_formats: List[str]

    # Analytics
    session_analytics: Dict[str, Any]

# Enhanced Role Detection with NLP
class AdvancedRoleDetector:
    """Advanced NLP-based role detection with specialized templates"""

    def __init__(self, nlp_model=None):
        self.nlp = nlp_model
        self.role_patterns = self._build_enhanced_patterns()
        self.skill_taxonomies = self._build_skill_taxonomies()

    def _build_enhanced_patterns(self) -> Dict[str, Dict]:
        """Enhanced role patterns with categories and confidence scoring"""
        return {
            # AI/ML Roles
            "founding_ai_engineer": {
                "patterns": ["founding.*ai", "founding.*ml", "founding.*artificial intelligence", "ai.*founding"],
                "category": "ai_ml",
                "base_role": "Founding AI Engineer",
                "confidence_boost": 0.2,
                "skills": ["Python", "PyTorch", "TensorFlow", "ML Pipeline", "System Design", "Leadership"],
                "specializations": ["Deep Learning", "MLOps", "AI Research", "Model Deployment"]
            },
            "genai_engineer": {
                "patterns": ["genai.*engineer", "generative.*ai.*engineer", "llm.*engineer", "ai.*engineer"],
                "category": "ai_ml",
                "base_role": "GenAI Engineer",
                "confidence_boost": 0.15,
                "skills": ["Python", "LangChain", "OpenAI API", "Vector Databases", "Prompt Engineering"],
                "specializations": ["LLM Fine-tuning", "RAG Systems", "AI Agents", "NLP"]
            },
            "genai_intern": {
                "patterns": ["genai.*intern", "ai.*intern", "ml.*intern", "generative.*ai.*intern"],
                "category": "ai_ml",
                "base_role": "GenAI Intern",
                "confidence_boost": 0.1,
                "skills": ["Python", "Machine Learning Basics", "Research", "Data Analysis"],
                "specializations": ["AI Research", "Model Training", "Data Science"]
            },
            "ml_engineer": {
                "patterns": ["ml.*engineer", "machine.*learning.*engineer", "ai.*ml.*engineer"],
                "category": "ai_ml",
                "base_role": "ML Engineer",
                "confidence_boost": 0.15,
                "skills": ["Python", "Scikit-learn", "MLflow", "Kubernetes", "AWS/GCP"],
                "specializations": ["MLOps", "Model Deployment", "Feature Engineering"]
            },
            "data_scientist": {
                "patterns": ["data.*scientist", "ds", "analytics.*scientist"],
                "category": "data",
                "base_role": "Data Scientist",
                "confidence_boost": 0.1,
                "skills": ["Python", "R", "SQL", "Statistics", "Pandas", "Jupyter"],
                "specializations": ["Statistical Modeling", "A/B Testing", "Business Intelligence"]
            },

            # Engineering Roles
            "founding_engineer": {
                "patterns": ["founding.*engineer", "founding.*developer", "technical.*cofounder"],
                "category": "engineering",
                "base_role": "Founding Engineer",
                "confidence_boost": 0.2,
                "skills": ["Python", "JavaScript", "System Design", "DevOps", "Leadership"],
                "specializations": ["Full-Stack", "Backend", "Infrastructure"]
            },
            "senior_engineer": {
                "patterns": ["senior.*engineer", "senior.*developer", "lead.*engineer"],
                "category": "engineering",
                "base_role": "Senior Software Engineer",
                "confidence_boost": 0.1,
                "skills": ["Python", "System Design", "Mentoring", "Architecture"],
                "specializations": ["Backend", "Frontend", "DevOps", "Mobile"]
            },
            "fullstack_engineer": {
                "patterns": ["fullstack.*engineer", "full.*stack.*engineer", "full.*stack.*developer"],
                "category": "engineering",
                "base_role": "Full-Stack Engineer",
                "confidence_boost": 0.1,
                "skills": ["JavaScript", "React", "Node.js", "Python", "Databases"],
                "specializations": ["Frontend", "Backend", "UI/UX"]
            },

            # Product & Design
            "product_manager": {
                "patterns": ["product.*manager", "pm", "product.*lead"],
                "category": "product",
                "base_role": "Product Manager",
                "confidence_boost": 0.1,
                "skills": ["Strategy", "Analytics", "User Research", "Roadmapping"],
                "specializations": ["B2B", "B2C", "Growth", "Technical PM"]
            },
            "ux_designer": {
                "patterns": ["ux.*designer", "ui.*ux.*designer", "user.*experience"],
                "category": "design",
                "base_role": "UX Designer",
                "confidence_boost": 0.1,
                "skills": ["Figma", "User Research", "Prototyping", "Design Systems"],
                "specializations": ["User Research", "Visual Design", "Interaction Design"]
            }
        }

    def _build_skill_taxonomies(self) -> Dict[str, List[str]]:
        """Build comprehensive skill taxonomies"""
        return {
            "ai_ml": {
                "frameworks": ["PyTorch", "TensorFlow", "Hugging Face", "LangChain", "LlamaIndex"],
                "languages": ["Python", "R", "Julia", "Scala"],
                "tools": ["Jupyter", "MLflow", "Weights & Biases", "Kubeflow", "Vector Databases"],
                "concepts": ["Deep Learning", "NLP", "Computer Vision", "MLOps", "Model Deployment"]
            },
            "engineering": {
                "languages": ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java"],
                "frontend": ["React", "Vue", "Angular", "Next.js", "Svelte"],
                "backend": ["Node.js", "Django", "FastAPI", "Spring", "Express"],
                "infrastructure": ["AWS", "GCP", "Azure", "Kubernetes", "Docker", "Terraform"]
            },
            "data": {
                "languages": ["Python", "R", "SQL", "Scala"],
                "tools": ["Pandas", "NumPy", "Spark", "Airflow", "dbt"],
                "visualization": ["Tableau", "PowerBI", "Plotly", "D3.js"],
                "databases": ["PostgreSQL", "MongoDB", "BigQuery", "Snowflake"]
            }
        }

    def detect_roles(self, text: str, company_context: Dict = None) -> List[EnhancedRoleRequirement]:
        """Enhanced role detection using NLP and pattern matching"""
        text_lower = text.lower()
        detected_roles = []

        # Use spaCy for entity recognition if available
        entities = []
        if self.nlp:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Pattern-based detection with confidence scoring
        for role_key, role_config in self.role_patterns.items():
            confidence = 0.0

            # Check pattern matches
            for pattern in role_config["patterns"]:
                if re.search(pattern, text_lower):
                    confidence += 0.7 + role_config.get("confidence_boost", 0)
                    break

            # Boost confidence based on context
            if company_context:
                if company_context.get("industry") == "AI/ML" and role_config["category"] == "ai_ml":
                    confidence += 0.1
                if company_context.get("stage") in ["Pre-seed", "Seed"] and "founding" in role_key:
                    confidence += 0.1

            # Add role if confidence threshold met
            if confidence >= 0.5:
                # Extract experience level
                technical_level = self._extract_experience_level(text_lower, role_key)

                # Extract timeline and budget hints
                timeline_days = self._extract_timeline(text)
                budget_range = self._extract_budget_hints(text)

                role = EnhancedRoleRequirement(
                    role_name=role_config["base_role"],
                    role_category=role_config["category"],
                    confidence_score=min(confidence, 1.0),
                    technical_level=technical_level,
                    specialization=role_config.get("specializations", []),
                    skills_required=role_config.get("skills", []),
                    timeline_days=timeline_days,
                    budget_min=budget_range.get("min") if budget_range else None,
                    budget_max=budget_range.get("max") if budget_range else None
                )

                detected_roles.append(role)

        # Sort by confidence and priority
        detected_roles.sort(key=lambda x: (x.confidence_score, -x.priority), reverse=True)

        # Default fallback
        if not detected_roles:
            detected_roles.append(EnhancedRoleRequirement(
                role_name="Software Engineer",
                role_category="engineering",
                confidence_score=0.3,
                skills_required=["Programming", "Problem Solving"]
            ))

        return detected_roles

    def _extract_experience_level(self, text: str, role_key: str) -> str:
        """Extract experience level from text"""
        if any(term in text for term in ["senior", "sr", "lead", "principal", "staff"]):
            return "senior"
        elif any(term in text for term in ["junior", "jr", "entry", "intern"]):
            return "junior"
        elif "founding" in role_key or "founding" in text:
            return "senior"  # Founding roles typically require senior experience
        else:
            return "mid"

    def _extract_timeline(self, text: str) -> Optional[int]:
        """Extract timeline from text"""
        # Look for timeline patterns
        timeline_patterns = [
            r"(\d+)\s*weeks?", r"(\d+)\s*months?", r"(\d+)\s*days?",
            r"asap", r"urgent", r"immediately"
        ]

        for pattern in timeline_patterns:
            match = re.search(pattern, text.lower())
            if match:
                if pattern in [r"asap", r"urgent", r"immediately"]:
                    return 14  # 2 weeks for urgent
                else:
                    num = int(match.group(1))
                    if "week" in pattern:
                        return num * 7
                    elif "month" in pattern:
                        return num * 30
                    else:
                        return num
        return None

    def _extract_budget_hints(self, text: str) -> Optional[Dict[str, int]]:
        """Extract budget hints from text"""
        # Look for budget patterns
        budget_pattern = r"\$(\d+)k?(?:\s*-\s*\$?(\d+)k?)?"
        matches = re.findall(budget_pattern, text)

        if matches:
            match = matches[0]
            min_val = int(match[0]) * 1000 if match[0] else None
            max_val = int(match[1]) * 1000 if match[1] else None

            if min_val and max_val:
                return {"min": min_val, "max": max_val}
            elif min_val:
                return {"min": min_val, "max": min_val * 1.3}  # Estimate max as 30% above min

        return None

print("✅ Enhanced data structures and NLP role detection ready!")

# ==================== CELL SEPARATOR ====================

class TavilySearchTool:
    """Tool 1: Enhanced Tavily Search for intelligent market research"""

    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced"
        ) if self.tavily_api_key else None
        self.cache = {}

    async def search_market_data(self, role: str, location: str = "United States", context: Dict = None) -> Dict[str, Any]:
        """Enhanced market research using Tavily's intelligent search"""
        cache_key = f"{role}_{location}_{context.get('industry', '') if context else ''}"

        if cache_key in self.cache:
            print(f"🧠 Using cached Tavily data for {role}")
            return self.cache[cache_key]

        print(f"🔍 Tavily search for {role} market data...")

        try:
            if self.search_tool and self.tavily_api_key != "your_tavily_api_key_here":
                # Enhanced search queries for better results
                queries = [
                    f"{role} salary range {location} 2024 startup",
                    f"{role} compensation trends {context.get('industry', 'technology') if context else 'technology'} companies",
                    f"{role} skills requirements job market analysis"
                ]

                search_results = {}
                for i, query in enumerate(queries):
                    try:
                        result = await self.search_tool.ainvoke({"query": query})
                        search_results[f"query_{i+1}"] = result
                    except Exception as e:
                        print(f"⚠️ Tavily search error for query {i+1}: {e}")

                # Process and synthesize results
                market_data = self._synthesize_search_results(search_results, role, context)

            else:
                # Enhanced mock data with more realistic ranges
                market_data = self._get_enhanced_mock_data(role, location, context)

        except Exception as e:
            print(f"⚠️ Tavily API error: {e}")
            market_data = self._get_enhanced_mock_data(role, location, context)

        # Cache results
        self.cache[cache_key] = market_data
        return market_data

    def _synthesize_search_results(self, search_results: Dict, role: str, context: Dict = None) -> Dict[str, Any]:
        """Synthesize Tavily search results into structured market data"""
        # This would process actual Tavily results
        # For now, return enhanced mock data structure
        return self._get_enhanced_mock_data(role, "United States", context)

    def _get_enhanced_mock_data(self, role: str, location: str, context: Dict = None) -> Dict[str, Any]:
        """Enhanced mock data with industry and company stage adjustments"""

        # Base salary database with more granular data
        base_salaries = {
            "founding ai engineer": {
                "base": {"min": 120000, "max": 200000, "median": 160000},
                "equity": "1.0% - 5.0%", "confidence": 0.9
            },
            "genai engineer": {
                "base": {"min": 130000, "max": 220000, "median": 175000},
                "equity": "0.1% - 1.0%", "confidence": 0.85
            },
            "genai intern": {
                "base": {"min": 25, "max": 45, "median": 35, "hourly": True},
                "equity": "0.01% - 0.1%", "confidence": 0.8
            },
            "ml engineer": {
                "base": {"min": 125000, "max": 210000, "median": 167500},
                "equity": "0.1% - 0.8%", "confidence": 0.85
            },
            "data scientist": {
                "base": {"min": 115000, "max": 190000, "median": 152500},
                "equity": "0.05% - 0.5%", "confidence": 0.8
            },
            "founding engineer": {
                "base": {"min": 110000, "max": 190000, "median": 150000},
                "equity": "0.5% - 3.0%", "confidence": 0.9
            },
            "senior software engineer": {
                "base": {"min": 140000, "max": 230000, "median": 185000},
                "equity": "0.05% - 0.3%", "confidence": 0.85
            },
            "product manager": {
                "base": {"min": 120000, "max": 200000, "median": 160000},
                "equity": "0.1% - 0.8%", "confidence": 0.8
            }
        }

        # Find matching role
        role_key = role.lower()
        salary_data = None

        for key, data in base_salaries.items():
            if key in role_key or any(word in role_key for word in key.split()):
                salary_data = data.copy()
                break

        if not salary_data:
            salary_data = {"base": {"min": 80000, "max": 150000, "median": 115000}, "equity": "0.1% - 0.5%", "confidence": 0.6}

        # Apply industry adjustments
        industry_multipliers = {
            "AI/ML": 1.2, "Technology": 1.1, "Finance": 1.15,
            "Healthcare": 1.0, "E-commerce": 1.05, "Other": 1.0
        }

        industry = context.get("industry", "Technology") if context else "Technology"
        multiplier = industry_multipliers.get(industry, 1.0)

        # Apply company stage adjustments
        stage_adjustments = {
            "Pre-seed": 0.85, "Seed": 0.9, "Series A": 1.0,
            "Series B": 1.1, "Series C+": 1.2
        }

        stage = context.get("stage", "Series A") if context else "Series A"
        stage_mult = stage_adjustments.get(stage, 1.0)

        # Calculate adjusted salaries
        if not salary_data["base"].get("hourly"):
            final_multiplier = multiplier * stage_mult
            salary_data["base"]["min"] = int(salary_data["base"]["min"] * final_multiplier)
            salary_data["base"]["max"] = int(salary_data["base"]["max"] * final_multiplier)
            salary_data["base"]["median"] = int(salary_data["base"]["median"] * final_multiplier)

        # Add market insights
        market_insights = {
            "market_temperature": "hot" if role_key in ["genai", "ai", "ml"] else "moderate",
            "hiring_difficulty": "high" if "founding" in role_key or "genai" in role_key else "medium",
            "time_to_hire_avg": "45-60 days" if "senior" in role_key or "founding" in role_key else "30-45 days",
            "top_skills_demand": self._get_trending_skills(role_key),
            "location_premium": 1.3 if location.lower() in ["san francisco", "new york", "seattle"] else 1.0
        }

        return {
            "tool": "tavily_search",
            "role": role,
            "location": location,
            "industry_context": industry,
            "company_stage": stage,
            "salary_data": salary_data,
            "market_insights": market_insights,
            "search_quality": "enhanced_mock",  # Would be "tavily_api" for real searches
            "last_updated": datetime.now().isoformat()
        }

    def _get_trending_skills(self, role_key: str) -> List[str]:
        """Get trending skills for role"""
        skill_trends = {
            "genai": ["LangChain", "Vector Databases", "Prompt Engineering", "RAG", "Fine-tuning"],
            "ai": ["PyTorch", "MLOps", "Model Deployment", "A/B Testing", "Data Pipeline"],
            "ml": ["MLflow", "Kubeflow", "Feature Engineering", "Model Monitoring", "AutoML"],
            "founding": ["System Design", "Leadership", "Full-Stack", "DevOps", "Product Strategy"],
            "senior": ["Architecture", "Mentoring", "Code Review", "Technical Leadership"],
            "data": ["SQL", "Python", "Statistical Modeling", "Business Intelligence", "ETL"]
        }

        for key, skills in skill_trends.items():
            if key in role_key:
                return skills

        return ["Programming", "Problem Solving", "Communication", "Collaboration"]

class EnhancedEmailWriterTool:
    """Tool 2: Enhanced Email Writer with role-specific templates"""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.role_templates = self._build_role_templates()
        self.email_cache = {}

    def _build_role_templates(self) -> Dict[str, Dict]:
        """Build specialized email templates by role category"""
        return {
            "ai_ml": {
                "subject_patterns": [
                    "Cutting-edge AI opportunity at {company}",
                    "Shape the future of AI at {company}",
                    "Join our AI innovation team at {company}"
                ],
                "intro_hooks": [
                    "We're building the next generation of AI systems",
                    "Your expertise in {specialization} caught our attention",
                    "We're looking for an AI innovator to join our founding team"
                ],
                "value_props": [
                    "Work on breakthrough AI research and applications",
                    "Access to latest models and unlimited compute resources",
                    "Collaborate with top AI researchers and engineers",
                    "Equity opportunity in high-growth AI startup"
                ]
            },
            "engineering": {
                "subject_patterns": [
                    "Engineering excellence opportunity at {company}",
                    "Build scalable systems at {company}",
                    "Senior engineering role at {company}"
                ],
                "intro_hooks": [
                    "We're scaling our engineering team",
                    "Your technical expertise aligns perfectly with our needs",
                    "Looking for a technical leader to help build our platform"
                ],
                "value_props": [
                    "Work on challenging technical problems at scale",
                    "Modern tech stack and engineering practices",
                    "High autonomy and technical decision-making",
                    "Competitive compensation and equity package"
                ]
            },
            "product": {
                "subject_patterns": [
                    "Product leadership opportunity at {company}",
                    "Drive product strategy at {company}",
                    "Join our product team at {company}"
                ],
                "value_props": [
                    "Own product strategy and roadmap",
                    "Work directly with founders and customers",
                    "High-impact role in growing startup",
                    "Equity opportunity with significant upside"
                ]
            }
        }

    async def generate_recruitment_email(
        self,
        role: EnhancedRoleRequirement,
        company_context: Dict,
        email_type: str = "outreach",
        market_data: Dict = None
    ) -> Dict[str, Any]:
        """Generate sophisticated recruitment emails using role-specific templates"""

        cache_key = f"{role.role_name}_{email_type}_{company_context.get('company_name', '')}"
        if cache_key in self.email_cache:
            return self.email_cache[cache_key]

        print(f"📧 Generating {email_type} email for {role.role_name}...")

        # Get role-specific template
        template = self.role_templates.get(role.role_category, self.role_templates["engineering"])

        # Build context for email generation
        email_context = {
            "role": role,
            "company": company_context,
            "market_data": market_data,
            "template": template,
            "email_type": email_type
        }

        try:
            # Generate email using enhanced prompt
            email_content = await self._generate_email_content(email_context)

            result = {
                "tool": "enhanced_email_writer",
                "email_type": email_type,
                "role_category": role.role_category,
                "subject": email_content.get("subject", ""),
                "body": email_content.get("body", ""),
                "personalization_notes": email_content.get("personalization", []),
                "generated_at": datetime.now().isoformat(),
                "template_used": role.role_category
            }

            self.email_cache[cache_key] = result
            return result

        except Exception as e:
            return {
                "tool": "enhanced_email_writer",
                "error": str(e),
                "fallback_subject": f"Exciting {role.role_name} Opportunity at {company_context.get('company_name', 'Our Company')}",
                "fallback_body": self._generate_fallback_email(role, company_context)
            }

    async def _generate_email_content(self, context: Dict) -> Dict[str, Any]:
        """Generate email content using AI with role-specific prompts"""

        role = context["role"]
        company = context["company"]
        template = context["template"]
        email_type = context["email_type"]
        market_data = context.get("market_data", {})

        # Build specialized prompt based on role category
        prompt = f"""
        Create a compelling {email_type} email for a {role.role_name} position.

        Role Context:
        - Position: {role.role_name}
        - Category: {role.role_category}
        - Technical Level: {role.technical_level}
        - Key Skills: {', '.join(role.skills_required[:5])}
        - Specializations: {', '.join(role.specialization[:3])}

        Company Context:
        - Name: {company.get('company_name', 'Our Company')}
        - Industry: {company.get('industry', 'Technology')}
        - Stage: {company.get('stage', 'Growth stage')}
        - Team Size: {company.get('team_size', '50')} people

        Market Context:
        - Market Temperature: {market_data.get('market_insights', {}).get('market_temperature', 'moderate')}
        - Hiring Difficulty: {market_data.get('market_insights', {}).get('hiring_difficulty', 'medium')}

        Email Type Specific Requirements:
        {self._get_email_type_requirements(email_type)}

        Use this role-specific template guidance:
        - Subject patterns: {template.get('subject_patterns', [])}
        - Value propositions: {template.get('value_props', [])}

        Generate:
        1. Subject line (compelling, specific to role)
        2. Email body (personalized, value-focused, 200-300 words)
        3. Personalization suggestions for recruiters

        Format as JSON with keys: subject, body, personalization
        """

        response = await self.openai_client.ainvoke([
            SystemMessage(content="You are an expert recruitment copywriter specializing in technical roles. Create engaging, personalized emails that highlight growth opportunities and technical challenges."),
            HumanMessage(content=prompt)
        ])

        # Parse response (in real implementation, would use structured output)
        content = response.content

        # Extract components (simplified parsing)
        lines = content.split('\n')
        subject = next((line.split(':', 1)[1].strip() for line in lines if line.lower().startswith('subject')), f"Exciting {role.role_name} Opportunity")

        # Find body content
        body_start = next((i for i, line in enumerate(lines) if 'body' in line.lower()), 0) + 1
        body_lines = lines[body_start:body_start+15]  # Take reasonable chunk
        body = '\n'.join(line for line in body_lines if line.strip() and not line.lower().startswith('personalization'))

        return {
            "subject": subject,
            "body": body or self._generate_fallback_email(role, company),
            "personalization": [
                f"Mention specific {role.role_category} projects",
                f"Reference candidate's experience with {role.skills_required[0] if role.skills_required else 'relevant technologies'}",
                "Highlight growth and learning opportunities"
            ]
        }

    def _get_email_type_requirements(self, email_type: str) -> str:
        """Get specific requirements for email type"""
        requirements = {
            "outreach": "Focus on opportunity and company growth. Include clear next steps.",
            "interview": "Confirm interview details. Build excitement. Include agenda and logistics.",
            "offer": "Present compelling offer package. Highlight total compensation and growth.",
            "rejection": "Be respectful and encouraging. Leave door open for future opportunities."
        }
        return requirements.get(email_type, "Professional and engaging communication.")

    def _generate_fallback_email(self, role: EnhancedRoleRequirement, company_context: Dict) -> str:
        """Generate fallback email if AI generation fails"""
        return f"""Hi [Candidate Name],

We have an exciting {role.role_name} opportunity at {company_context.get('company_name', 'our company')} that would be perfect for someone with your {role.role_category} background.

As a {company_context.get('stage', 'growing')} {company_context.get('industry', 'technology')} company, we're looking for talented individuals who want to make a significant impact while working with cutting-edge technology.

Key highlights:
• Work with {', '.join(role.skills_required[:3])} and other modern technologies
• High-growth environment with learning opportunities
• Competitive compensation including equity
• Collaborative team of {company_context.get('team_size', '20+')} talented individuals

Would you be interested in learning more about this opportunity?

Best regards,
[Your Name]"""

class EnhancedChecklistBuilderTool:
    """Tool 3: Enhanced Checklist Builder with adaptive workflows"""

    def __init__(self):
        self.workflow_templates = self._build_workflow_templates()
        self.checklist_cache = {}

    def _build_workflow_templates(self) -> Dict[str, Dict]:
        """Build adaptive workflow templates by role complexity"""
        return {
            "ai_ml_roles": {
                "complexity_score": 0.9,
                "additional_phases": ["Technical Deep Dive", "AI/ML Assessment", "Research Presentation"],
                "specialized_tasks": [
                    "Review AI/ML portfolio and research papers",
                    "Conduct technical interview on model architecture",
                    "Assess knowledge of latest AI developments",
                    "Evaluate coding skills in Python/PyTorch",
                    "Test understanding of MLOps and deployment"
                ]
            },
            "founding_roles": {
                "complexity_score": 0.95,
                "additional_phases": ["Leadership Assessment", "Vision Alignment", "Equity Discussion"],
                "specialized_tasks": [
                    "Assess technical leadership experience",
                    "Evaluate startup and scaling experience",
                    "Test system design and architecture skills",
                    "Assess cultural fit and values alignment",
                    "Discuss equity expectations and vesting"
                ]
            },
            "senior_roles": {
                "complexity_score": 0.8,
                "additional_phases": ["Technical Leadership", "Mentoring Assessment"],
                "specialized_tasks": [
                    "Evaluate technical mentoring experience",
                    "Assess code review and architecture skills",
                    "Test cross-functional collaboration",
                    "Review past technical decisions and learnings"
                ]
            },
            "standard_roles": {
                "complexity_score": 0.6,
                "additional_phases": [],
                "specialized_tasks": [
                    "Standard technical assessment",
                    "Team collaboration evaluation",
                    "Culture fit interview"
                ]
            }
        }

    def create_adaptive_checklist(
        self,
        roles: List[EnhancedRoleRequirement],
        company_context: Dict,
        market_data: Dict = None
    ) -> Dict[str, Any]:
        """Create adaptive hiring checklist based on role complexity and market conditions"""

        cache_key = f"{len(roles)}_{hash(str([r.role_name for r in roles]))}"
        if cache_key in self.checklist_cache:
            return self.checklist_cache[cache_key]

        print(f"📋 Building adaptive checklist for {len(roles)} roles...")

        # Analyze role complexity
        complexity_analysis = self._analyze_role_complexity(roles)

        # Determine base timeline
        base_timeline = self._calculate_adaptive_timeline(roles, market_data)

        # Build checklist
        checklist = {
            "tool": "enhanced_checklist_builder",
            "created_at": datetime.now().isoformat(),
            "roles_analysis": complexity_analysis,
            "adaptive_timeline": base_timeline,
            "total_phases": 0,
            "phases": []
        }

        # Phase 1: Enhanced Preparation
        checklist["phases"].append(self._create_preparation_phase(roles, complexity_analysis))

        # Phase 2: Intelligent Sourcing
        checklist["phases"].append(self._create_sourcing_phase(roles, company_context, market_data))

        # Phase 3: Role-specific Screening
        for role in roles:
            checklist["phases"].append(self._create_role_screening_phase(role, complexity_analysis))

        # Phase 4: Advanced Assessment
        checklist["phases"].append(self._create_assessment_phase(roles, complexity_analysis))

        # Phase 5: Decision & Closing
        checklist["phases"].append(self._create_closing_phase(roles, market_data))

        # Add success metrics and risk mitigation
        checklist.update({
            "total_phases": len(checklist["phases"]),
            "success_metrics": self._generate_success_metrics(roles, market_data),
            "risk_mitigation": self._generate_risk_mitigation(roles, complexity_analysis),
            "adaptive_features": self._generate_adaptive_features(complexity_analysis)
        })

        self.checklist_cache[cache_key] = checklist
        return checklist

    def _analyze_role_complexity(self, roles: List[EnhancedRoleRequirement]) -> Dict[str, Any]:
        """Analyze complexity of roles for adaptive planning"""
        complexity_scores = []
        role_categories = set()
        special_requirements = []

        for role in roles:
            # Get complexity template
            if "founding" in role.role_name.lower():
                template_key = "founding_roles"
            elif role.role_category == "ai_ml":
                template_key = "ai_ml_roles"
            elif role.technical_level == "senior":
                template_key = "senior_roles"
            else:
                template_key = "standard_roles"

            template = self.workflow_templates[template_key]
            complexity_scores.append(template["complexity_score"])
            role_categories.add(role.role_category)
            special_requirements.extend(template["specialized_tasks"])

        return {
            "average_complexity": sum(complexity_scores) / len(complexity_scores),
            "max_complexity": max(complexity_scores),
            "unique_categories": len(role_categories),
            "special_requirements": list(set(special_requirements)),
            "requires_specialized_assessment": any(score > 0.8 for score in complexity_scores)
        }

    def _calculate_adaptive_timeline(self, roles: List[EnhancedRoleRequirement], market_data: Dict = None) -> int:
        """Calculate adaptive timeline based on role complexity and market conditions"""

        base_days = 30  # Standard baseline

        # Adjust for role complexity
        complexity_adjustment = 0
        for role in roles:
            if "founding" in role.role_name.lower():
                complexity_adjustment += 14  # Founding roles take longer
            elif role.role_category == "ai_ml":
                complexity_adjustment += 7   # AI/ML roles need more assessment
            elif role.technical_level == "senior":
                complexity_adjustment += 5   # Senior roles need more vetting

        # Adjust for market conditions
        market_adjustment = 0
        if market_data:
            for role_name, data in market_data.items():
                insights = data.get('market_insights', {})
                if insights.get('market_temperature') == 'hot':
                    market_adjustment += 7  # Hot market = longer search
                if insights.get('hiring_difficulty') == 'high':
                    market_adjustment += 5  # Difficult roles take longer

        # Calculate final timeline
        total_timeline = base_days + complexity_adjustment + market_adjustment

        # Cap at reasonable maximums
        return min(total_timeline, 90)  # Max 90 days

    def _create_preparation_phase(self, roles: List[EnhancedRoleRequirement], complexity: Dict) -> Dict[str, Any]:
        """Create enhanced preparation phase"""
        base_tasks = [
            "✅ Finalize role-specific job descriptions with market positioning",
            "✅ Set up intelligent hiring pipeline with candidate scoring",
            "✅ Prepare adaptive interview kits by role complexity",
            "✅ Define success criteria and evaluation rubrics",
            "✅ Configure applicant tracking and candidate experience tools"
        ]

        # Add complexity-specific tasks
        if complexity["requires_specialized_assessment"]:
            base_tasks.extend([
                "✅ Prepare specialized technical assessments",
                "✅ Brief interviewers on advanced evaluation criteria",
                "✅ Set up coding challenge environments"
            ])

        return {
            "phase": "1. Enhanced Preparation",
            "duration_days": 4 if complexity["requires_specialized_assessment"] else 3,
            "tasks": base_tasks,
            "deliverables": ["Job descriptions", "Interview kits", "Assessment criteria", "Scoring rubrics"],
            "owner": "Hiring Manager + HR",
            "success_criteria": "All preparation materials ready and team trained on evaluation process"
        }

    def _create_sourcing_phase(self, roles: List[EnhancedRoleRequirement], company_context: Dict, market_data: Dict) -> Dict[str, Any]:
        """Create intelligent sourcing phase"""
        base_tasks = [
            "✅ Multi-platform job posting with role-optimized descriptions",
            "✅ Targeted sourcing using role-specific keywords and communities",
            "✅ Employee referral program with incentive structure",
            "✅ Direct outreach via LinkedIn and specialized platforms",
            "✅ University partnerships and campus recruiting (if applicable)"
        ]

        # Add role-specific sourcing strategies
        ai_ml_roles = [r for r in roles if r.role_category == "ai_ml"]
        if ai_ml_roles:
            base_tasks.extend([
                "✅ Source from AI/ML conferences and research communities",
                "✅ Engage with AI research labs and universities",
                "✅ Post in specialized AI/ML job boards and communities"
            ])

        sourcing_duration = max(10, len(roles) * 3)  # Scale with role count

        return {
            "phase": "2. Intelligent Sourcing",
            "duration_days": sourcing_duration,
            "tasks": base_tasks,
            "deliverables": ["Candidate pipeline", "Sourcing analytics", "Response rates by channel"],
            "owner": "Recruitment Team + Hiring Managers",
            "success_criteria": f"3:1 qualified candidate ratio for each role"
        }

    def _create_role_screening_phase(self, role: EnhancedRoleRequirement, complexity: Dict) -> Dict[str, Any]:
        """Create role-specific screening phase"""
        base_tasks = [
            "✅ Resume and portfolio initial screening",
            "✅ Phone/video screening interview (30-45 min)",
            "✅ Basic skills and motivation assessment",
            "✅ Cultural fit and communication evaluation"
        ]

        # Add role-specific screening tasks
        if role.role_category == "ai_ml":
            base_tasks.extend([
                "✅ Review AI/ML projects and research contributions",
                "✅ Assess technical knowledge in machine learning",
                "✅ Evaluate experience with relevant frameworks"
            ])
        elif "founding" in role.role_name.lower():
            base_tasks.extend([
                "✅ Assess startup and leadership experience",
                "✅ Evaluate technical vision and strategic thinking",
                "✅ Review past team building and scaling experience"
            ])

        return {
            "phase": f"3. Screening - {role.role_name}",
            "duration_days": 5,
            "tasks": base_tasks,
            "deliverables": ["Shortlisted candidates", "Screening scorecards", "Interview notes"],
            "owner": "Hiring Manager",
            "success_criteria": "3-5 qualified candidates advance to technical assessment"
        }

    def _create_assessment_phase(self, roles: List[EnhancedRoleRequirement], complexity: Dict) -> Dict[str, Any]:
        """Create advanced assessment phase"""
        base_tasks = [
            "✅ Technical assessment and coding challenges",
            "✅ System design or case study evaluation",
            "✅ Team collaboration and communication assessment",
            "✅ Final stakeholder interviews"
        ]

        # Add specialized assessments
        if complexity["requires_specialized_assessment"]:
            base_tasks.extend([
                "✅ Role-specific deep-dive technical sessions",
                "✅ Presentation of past work or research",
                "✅ Problem-solving and architecture discussions"
            ])

        return {
            "phase": "4. Advanced Assessment",
            "duration_days": 7,
            "tasks": base_tasks,
            "deliverables": ["Technical assessments", "Interview feedback", "Final candidate rankings"],
            "owner": "Full Interview Panel",
            "success_criteria": "Clear top choice identified for each role with consensus"
        }

    def _create_closing_phase(self, roles: List[EnhancedRoleRequirement], market_data: Dict) -> Dict[str, Any]:
        """Create optimized closing phase"""
        base_tasks = [
            "✅ Comprehensive reference checks",
            "✅ Final team consensus and approval",
            "✅ Competitive offer package preparation",
            "✅ Offer presentation and negotiation",
            "✅ Contract finalization and onboarding coordination"
        ]

        # Add market-specific tasks
        if market_data:
            hot_market_roles = any(
                data.get('market_insights', {}).get('market_temperature') == 'hot'
                for data in market_data.values()
            )
            if hot_market_roles:
                base_tasks.insert(2, "✅ Expedited decision process for hot market roles")

        return {
            "phase": "5. Decision & Closing",
            "duration_days": 7,
            "tasks": base_tasks,
            "deliverables": ["Signed offers", "Background checks", "Onboarding schedule"],
            "owner": "Hiring Manager + HR + Legal",
            "success_criteria": "90%+ offer acceptance rate with smooth onboarding transition"
        }

    def _generate_success_metrics(self, roles: List[EnhancedRoleRequirement], market_data: Dict) -> List[str]:
        """Generate role-specific success metrics"""
        base_metrics = [
            "Quality of hire score > 4.5/5 after 90 days",
            "Candidate experience rating > 4.0/5",
            "Diversity representation meets company targets",
            "Budget adherence within 10% of planned spend"
        ]

        # Add role-specific metrics
        if any(r.role_category == "ai_ml" for r in roles):
            base_metrics.append("AI/ML candidates demonstrate practical model deployment experience")

        if any("founding" in r.role_name.lower() for r in roles):
            base_metrics.append("Founding hires show measurable impact on product/team within 60 days")

        return base_metrics

    def _generate_risk_mitigation(self, roles: List[EnhancedRoleRequirement], complexity: Dict) -> List[str]:
        """Generate adaptive risk mitigation strategies"""
        base_strategies = [
            "Maintain 3:1 candidate pipeline ratio minimum",
            "Pre-approve competitive salary bands and equity ranges",
            "Build 20% timeline buffer for unexpected delays",
            "Prepare compelling employer brand materials"
        ]

        # Add complexity-specific strategies
        if complexity["requires_specialized_assessment"]:
            base_strategies.extend([
                "Have backup technical interviewers trained and available",
                "Prepare multiple assessment formats for different candidate preferences",
                "Establish partnerships with technical recruiting specialists"
            ])

        return base_strategies

    def _generate_adaptive_features(self, complexity: Dict) -> List[str]:
        """Generate adaptive workflow features"""
        features = [
            "Dynamic timeline adjustment based on candidate availability",
            "Automated candidate scoring and ranking",
            "Real-time pipeline analytics and bottleneck detection"
        ]

        if complexity["requires_specialized_assessment"]:
            features.extend([
                "Specialized technical assessment routing",
                "Expert interviewer allocation based on role requirements",
                "Advanced candidate portfolio review workflows"
            ])

        return features

print("✅ Enhanced three core tools with Tavily integration complete!")

# ==================== CELL SEPARATOR ====================

# ================================
# CELL 4: Enhanced Multi-Agent System with Intent Classification & Conditional Routing
# ================================

class IntentClassifier:
    """Classify user intent to determine workflow routing"""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.conversation_patterns = [
            "hi", "hello", "hey", "how are you", "what can you do",
            "help", "capabilities", "features", "how does this work",
            "what is this", "explain", "about", "info"
        ]
        self.hiring_patterns = [
            "hire", "hiring", "recruit", "find", "need", "looking for",
            "position", "role", "job", "engineer", "developer", "manager",
            "intern", "candidate", "team", "staff"
        ]

    async def classify_intent(self, user_input: str) -> Dict[str, Any]:
        """Classify user intent with confidence scoring"""

        user_lower = user_input.lower().strip()

        # Quick pattern matching for obvious cases
        conversation_score = sum(1 for pattern in self.conversation_patterns if pattern in user_lower)
        hiring_score = sum(1 for pattern in self.hiring_patterns if pattern in user_lower)

        # Simple heuristics first
        if len(user_input.split()) <= 5 and conversation_score > 0 and hiring_score == 0:
            return {
                "intent": "conversation",
                "confidence": 0.9,
                "reasoning": "Short conversational query",
                "requires_tools": False
            }

        if hiring_score >= 2 or any(role in user_lower for role in ["engineer", "developer", "intern", "manager"]):
            return {
                "intent": "hiring",
                "confidence": 0.8 + (hiring_score * 0.1),
                "reasoning": "Contains hiring-related keywords",
                "requires_tools": True
            }

        # Use AI for ambiguous cases
        if conversation_score > 0 and hiring_score > 0:
            return await self._ai_classify(user_input)

        # Default to conversation for unclear inputs
        return {
            "intent": "conversation",
            "confidence": 0.6,
            "reasoning": "Unclear intent, defaulting to conversation",
            "requires_tools": False
        }

    async def _ai_classify(self, user_input: str) -> Dict[str, Any]:
        """Use AI for complex intent classification"""

        prompt = f"""
        Classify this user input into one of two intents:

        1. "conversation" - General questions, greetings, help requests, capability inquiries
        2. "hiring" - Requests to help with recruiting, hiring, finding candidates, job descriptions

        User input: "{user_input}"

        Respond with just: conversation OR hiring
        """

        try:
            response = await self.openai_client.ainvoke([
                SystemMessage(content="You are an intent classifier. Respond with only 'conversation' or 'hiring'."),
                HumanMessage(content=prompt)
            ])

            intent = response.content.strip().lower()
            if intent not in ["conversation", "hiring"]:
                intent = "conversation"

            return {
                "intent": intent,
                "confidence": 0.7,
                "reasoning": "AI classification",
                "requires_tools": intent == "hiring"
            }
        except Exception:
            return {
                "intent": "conversation",
                "confidence": 0.5,
                "reasoning": "AI classification failed, defaulting to conversation",
                "requires_tools": False
            }

class ConversationalAgent:
    """Handle non-hiring conversational queries"""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.capabilities = [
            "🎯 **Role Detection**: Identify hiring needs with advanced NLP",
            "🔍 **Market Research**: Get salary data and market insights via Tavily Search",
            "📝 **Content Generation**: Create job descriptions and recruitment emails",
            "📋 **Workflow Planning**: Build adaptive hiring checklists and timelines",
            "💾 **Session Memory**: Remember context across conversations",
            "📊 **Analytics**: Track hiring metrics and success patterns",
            "📤 **Export**: Generate professional reports in JSON/Markdown"
        ]

    async def generate_response(self, user_input: str, context: Dict = None) -> str:
        """Generate conversational response"""

        user_lower = user_input.lower()

        # Handle common greetings and help requests
        if any(greeting in user_lower for greeting in ["hi", "hello", "hey"]):
            return self._greeting_response()

        if any(help_word in user_lower for help_word in ["help", "what can you do", "capabilities"]):
            return self._capabilities_response()

        if any(word in user_lower for word in ["how", "work", "explain"]):
            return self._explanation_response()

        # Use AI for complex conversational queries
        return await self._ai_conversational_response(user_input)

    def _greeting_response(self) -> str:
        return """👋 **Hello! I'm your AI HR Agent assistant.**

I specialize in helping startups plan their hiring process. Here's how I can help you:

🎯 **Smart Hiring Analysis**
- Detect roles from natural language descriptions
- Analyze market conditions and salary ranges
- Generate confidence scores for recommendations

🔧 **Automated Content Creation**
- AI-generated job descriptions tailored to your company
- Professional recruitment email templates
- Adaptive hiring workflows and checklists

💡 **Intelligent Insights**
- Market research via advanced search capabilities
- Role-specific skill requirements and trends
- Timeline optimization based on role complexity

**To get started, just describe your hiring needs!** For example:
- "I need to hire a founding engineer and GenAI intern"
- "Help me find a senior data scientist with ML experience"
- "We need a product manager for our fintech startup"

What hiring challenge can I help you solve today? 🚀"""

    def _capabilities_response(self) -> str:
        capabilities_text = "🤖 **My Capabilities:**\n\n" + "\n".join(self.capabilities)

        capabilities_text += """

🎯 **How to Use Me:**
1. **Describe your hiring needs** in natural language
2. **I'll analyze** roles, skills, and requirements
3. **Get comprehensive results** including job descriptions, workflows, and market data
4. **Export professional reports** for your team

**Example requests:**
- "I need to hire a founding engineer and GenAI intern"
- "Help me find a senior data scientist, budget 140k-160k"
- "We need a product manager for our healthcare startup"

Ready to streamline your hiring process? Just tell me what roles you need! 🚀"""

        return capabilities_text

    def _explanation_response(self) -> str:
        return """🔧 **How I Work:**

**1. Intent Analysis** 🧠
- I first understand if you're asking about hiring or just having a conversation
- Only activate advanced tools when needed for efficiency

**2. Smart Role Detection** 🎯
- Use advanced NLP to identify specific roles and requirements
- Generate confidence scores for each detected position
- Ask clarifying questions only when necessary

**3. Multi-Agent Coordination** 🤖
- Research Agent: Gathers market data via Tavily Search
- Content Agent: Creates job descriptions and email templates
- Integration Agent: Builds adaptive hiring workflows
- Memory Agent: Maintains conversation context

**4. Intelligent Tool Selection** ⚙️
- **Tavily Search**: For market research and salary data
- **Email Writer**: For recruitment communication templates
- **Checklist Builder**: For hiring workflow automation

**5. Professional Output** 📊
- Generate comprehensive hiring plans
- Export results in JSON/Markdown formats
- Provide actionable next steps

Ready to see it in action? Describe a hiring need and I'll demonstrate! 🚀"""

    async def _ai_conversational_response(self, user_input: str) -> str:
        """Generate AI-powered conversational response"""

        prompt = f"""
        You are a helpful AI HR Assistant. The user asked: "{user_input}"

        This is NOT a hiring request - it's a conversational question. Provide a helpful, friendly response that:
        1. Addresses their question directly
        2. Relates to HR/hiring topics when relevant
        3. Encourages them to try hiring-related features
        4. Keeps tone professional but approachable
        5. Is concise (under 200 words)

        Do NOT attempt to detect roles or create hiring plans.
        """

        try:
            response = await self.openai_client.ainvoke([
                SystemMessage(content="You are a helpful AI HR Assistant focused on conversational support."),
                HumanMessage(content=prompt)
            ])
            return response.content
        except Exception:
            return """I'm here to help with your hiring needs!

While I didn't quite understand your question, I specialize in:
- Analyzing hiring requirements
- Generating job descriptions
- Creating hiring workflows
- Providing market insights

Try asking me about a specific role you need to hire for! 🚀"""

class EnhancedReactAgent:
    """Enhanced individual agent with conditional processing based on intent"""

    def __init__(self, role: AgentRole, tools: Dict[str, Any]):
        self.role = role
        self.tools = tools
        self.agent_memory = {}
        self.performance_history = []

    async def process_with_enhanced_reasoning(self, state: EnhancedHRAgentState, task: str) -> Dict[str, Any]:
        """Enhanced processing with ReAct patterns and conditional execution"""

        start_time = datetime.now()

        # Check if this agent should run based on intent
        intent_data = state.get("intent_classification", {})
        if not self._should_process(intent_data, task):
            return {
                "agent": self.role.value,
                "skipped": True,
                "reasoning": f"Skipped - not needed for {intent_data.get('intent', 'unknown')} intent",
                "execution_time": 0,
                "success": True
            }

        # Reasoning phase
        reasoning = await self._enhanced_reasoning(state, task)

        # Acting phase (only if tools are needed)
        if intent_data.get("requires_tools", True):
            action_result = await self._specialized_action(state, task, reasoning)
        else:
            action_result = {"success": True, "skipped_tools": True, "reasoning": "No tools needed for conversational intent"}

        # Reflecting phase
        reflection = await self._reflection_and_learning(reasoning, action_result, state)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Store performance
        performance = {
            "agent": self.role.value,
            "task": task,
            "execution_time": execution_time,
            "success": action_result.get("success", False),
            "confidence": reasoning.get("confidence", 0.5),
            "timestamp": datetime.now()
        }
        self.performance_history.append(performance)

        return {
            "agent": self.role.value,
            "reasoning": reasoning,
            "action": action_result,
            "reflection": reflection,
            "performance": performance
        }

    def _should_process(self, intent_data: Dict, task: str) -> bool:
        """Determine if this agent should process based on intent"""

        intent = intent_data.get("intent", "hiring")
        requires_tools = intent_data.get("requires_tools", True)

        # Skip most agents for conversational intent
        if intent == "conversation":
            return self.role in [AgentRole.COORDINATOR, AgentRole.OUTPUT_FORMATTER]

        # For hiring intent, all agents can process
        return True

    async def _enhanced_reasoning(self, state: EnhancedHRAgentState, task: str) -> Dict[str, Any]:
        """Enhanced reasoning with agent specialization and intent awareness"""

        intent_data = state.get("intent_classification", {})

        context = {
            "current_stage": state.get("workflow_stage", "unknown"),
            "detected_roles": len(state.get("detected_roles", [])),
            "session_type": "followup" if state.get("is_followup", False) else "new",
            "agent_specialization": self.role.value,
            "intent": intent_data.get("intent", "hiring"),
            "requires_tools": intent_data.get("requires_tools", True)
        }

        # Agent-specific reasoning with intent consideration
        if self.role == AgentRole.NLP_ANALYZER:
            reasoning_focus = "role detection and requirement extraction" if context["intent"] == "hiring" else "conversational understanding"
        elif self.role == AgentRole.RESEARCH:
            reasoning_focus = "market research and competitive intelligence"
        elif self.role == AgentRole.CONTENT:
            reasoning_focus = "content generation and template optimization"
        elif self.role == AgentRole.MEMORY_MANAGER:
            reasoning_focus = "session continuity and change detection"
        else:
            reasoning_focus = "coordination and workflow optimization"

        return {
            "agent": self.role.value,
            "focus": reasoning_focus,
            "context": context,
            "confidence": 0.8,
            "reasoning_text": f"As {self.role.value}, focusing on {reasoning_focus} for {task} (intent: {context['intent']})",
            "timestamp": datetime.now()
        }

    async def _specialized_action(self, state: EnhancedHRAgentState, task: str, reasoning: Dict) -> Dict[str, Any]:
        """Execute specialized actions based on agent role with intent awareness"""

        try:
            if self.role == AgentRole.NLP_ANALYZER:
                return await self._nlp_analysis_action(state)
            elif self.role == AgentRole.RESEARCH:
                return await self._research_action(state)
            elif self.role == AgentRole.CONTENT:
                return await self._content_generation_action(state)
            elif self.role == AgentRole.INTEGRATION:
                return await self._integration_planning_action(state)
            elif self.role == AgentRole.MEMORY_MANAGER:
                return await self._memory_management_action(state)
            elif self.role == AgentRole.OUTPUT_FORMATTER:
                return await self._output_formatting_action(state)
            else:
                return await self._coordination_action(state, task)

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "agent": self.role.value,
                "fallback_executed": True
            }

    async def _nlp_analysis_action(self, state: EnhancedHRAgentState) -> Dict[str, Any]:
        """NLP-based role detection and analysis"""

        messages = state.get("messages", [])
        last_message = messages[-1].content if messages else ""
        company_context = state.get("company_context", {})

        # Use enhanced role detector (can be simplified without spaCy if needed)
        role_detector = self.tools.get("role_detector")
        if role_detector:
            detected_roles = role_detector.detect_roles(last_message, company_context)
        else:
            # Fallback simple detection
            detected_roles = self._simple_role_detection(last_message)

        # Analyze confidence and generate clarifications
        clarifications = []
        total_confidence = sum(role.confidence_score for role in detected_roles) / len(detected_roles) if detected_roles else 0.5

        if total_confidence < 0.7:
            clarifications.append("Could you provide more details about the specific roles you need?")

        # Check for missing information
        for role in detected_roles:
            if not role.budget_min and not role.budget_max:
                clarifications.append(f"What's your budget range for the {role.role_name} position?")
            if not role.timeline_days:
                clarifications.append(f"What's your ideal timeline for hiring the {role.role_name}?")
            if not role.skills_required:
                clarifications.append(f"Are there specific skills or technologies required for the {role.role_name}?")

        return {
            "success": True,
            "detected_roles": [asdict(role) for role in detected_roles],
            "confidence": total_confidence,
            "clarifications_needed": clarifications,
            "analysis_type": "enhanced_nlp",
            "roles_count": len(detected_roles)
        }

    def _simple_role_detection(self, text: str) -> List[EnhancedRoleRequirement]:
        """Simple role detection fallback without spaCy dependency"""

        roles = []
        text_lower = text.lower()

        # Define role patterns
        role_patterns = {
            "founding engineer": {"category": "engineering", "skills": ["Python", "System Design", "Leadership"]},
            "genai engineer": {"category": "ai_ml", "skills": ["Python", "ML", "LangChain"]},
            "genai intern": {"category": "ai_ml", "skills": ["Python", "Research"]},
            "senior engineer": {"category": "engineering", "skills": ["Python", "Architecture"]},
            "data scientist": {"category": "data", "skills": ["Python", "Statistics"]},
            "product manager": {"category": "product", "skills": ["Strategy", "Analytics"]}
        }

        for pattern, config in role_patterns.items():
            if pattern in text_lower:
                role = EnhancedRoleRequirement(
                    role_name=pattern.title(),
                    role_category=config["category"],
                    confidence_score=0.8,
                    skills_required=config["skills"]
                )
                roles.append(role)

        # Default fallback
        if not roles:
            roles.append(EnhancedRoleRequirement(
                role_name="Software Engineer",
                role_category="engineering",
                confidence_score=0.5,
                skills_required=["Programming"]
            ))

        return roles

    async def _research_action(self, state: EnhancedHRAgentState) -> Dict[str, Any]:
        """Enhanced market research using Tavily"""

        detected_roles = state.get("detected_roles", [])
        company_context = state.get("company_context", {})

        research_results = {}

        for role_data in detected_roles:
            role_name = role_data.get("role_name", "")

            # Use Tavily search tool
            tavily_tool = self.tools["tavily_search"]
            market_data = await tavily_tool.search_market_data(
                role_name,
                company_context.get("location", "United States"),
                company_context
            )

            research_results[role_name] = market_data

        # Generate market insights summary
        insights = self._generate_market_insights(research_results)

        return {
            "success": True,
            "research_data": research_results,
            "market_insights": insights,
            "research_quality": "tavily_enhanced",
            "roles_researched": len(detected_roles)
        }

    async def _content_generation_action(self, state: EnhancedHRAgentState) -> Dict[str, Any]:
        """Enhanced content generation with role specialization"""

        detected_roles = state.get("detected_roles", [])
        company_context = state.get("company_context", {})
        research_data = state.get("agent_outputs", {}).get("research", {}).get("research_data", {})

        content_results = {
            "job_descriptions": {},
            "email_templates": {},
            "content_quality": "enhanced"
        }

        email_writer = self.tools["email_writer"]

        for role_data in detected_roles:
            role_obj = EnhancedRoleRequirement(**role_data)
            role_name = role_obj.role_name

            # Generate job description using AI
            job_desc = await self._generate_enhanced_job_description(
                role_obj, company_context, research_data.get(role_name, {})
            )
            content_results["job_descriptions"][role_name] = job_desc

            # Generate email templates
            outreach_email = await email_writer.generate_recruitment_email(
                role_obj, company_context, "outreach", research_data.get(role_name, {})
            )
            interview_email = await email_writer.generate_recruitment_email(
                role_obj, company_context, "interview", research_data.get(role_name, {})
            )

            content_results["email_templates"][role_name] = {
                "outreach": outreach_email,
                "interview": interview_email
            }

        return {
            "success": True,
            "content_deliverables": content_results,
            "generation_method": "ai_enhanced",
            "roles_processed": len(detected_roles)
        }

    async def _integration_planning_action(self, state: EnhancedHRAgentState) -> Dict[str, Any]:
        """Enhanced integration and workflow planning"""

        detected_roles = state.get("detected_roles", [])
        company_context = state.get("company_context", {})
        research_data = state.get("agent_outputs", {}).get("research", {}).get("research_data", {})

        # Convert to EnhancedRoleRequirement objects
        role_objects = [EnhancedRoleRequirement(**role_data) for role_data in detected_roles]

        # Use enhanced checklist builder
        checklist_builder = self.tools["checklist_builder"]
        hiring_checklist = checklist_builder.create_adaptive_checklist(
            role_objects, company_context, research_data
        )

        # Generate integration recommendations
        integration_recommendations = self._generate_integration_recommendations(
            role_objects, company_context, research_data
        )

        return {
            "success": True,
            "hiring_checklist": hiring_checklist,
            "integration_recommendations": integration_recommendations,
            "workflow_type": "adaptive",
            "complexity_score": hiring_checklist.get("roles_analysis", {}).get("average_complexity", 0.6)
        }

    async def _memory_management_action(self, state: EnhancedHRAgentState) -> Dict[str, Any]:
        """Enhanced memory and session management"""

        session_id = state.get("session_id", "")
        conversation_memory = state.get("conversation_memory", {})
        is_followup = state.get("is_followup", False)

        # Detect changes if this is a followup
        change_analysis = {}
        if is_followup:
            change_analysis = self._analyze_conversation_changes(state)

        # Update memory
        updated_memory = {
            "session_id": session_id,
            "last_updated": datetime.now().isoformat(),
            "conversation_count": conversation_memory.get("conversation_count", 0) + 1,
            "change_analysis": change_analysis,
            "user_preferences": self._extract_user_preferences(state),
            "company_context": state.get("company_context", {})
        }

        return {
            "success": True,
            "memory_update": updated_memory,
            "is_followup": is_followup,
            "changes_detected": len(change_analysis.get("changes", [])),
            "memory_quality": "enhanced"
        }

    async def _output_formatting_action(self, state: EnhancedHRAgentState) -> Dict[str, Any]:
        """Enhanced output formatting for both conversation and hiring"""

        intent = state.get("intent_classification", {}).get("intent", "hiring")
        agent_outputs = state.get("agent_outputs", {})

        if intent == "conversation":
            # Simple conversational output formatting
            conversation_data = agent_outputs.get("conversation", {})

            structured_output = {
                "type": "conversation",
                "response": conversation_data.get("response", ""),
                "intent": intent,
                "tools_used": [],
                "session_id": state.get("session_id", ""),
                "generated_at": datetime.now().isoformat()
            }
        else:
            # Complex hiring output formatting
            structured_output = {
                "hiring_plan": {
                    "roles": state.get("detected_roles", []),
                    "market_research": agent_outputs.get("research", {}),
                    "content_deliverables": agent_outputs.get("content", {}),
                    "workflow": agent_outputs.get("integration", {}),
                    "session_info": agent_outputs.get("memory", {})
                },
                "export_formats": {
                    "json": True,
                    "markdown": True,
                    "pdf": True
                },
                "generated_at": datetime.now().isoformat()
            }

        # Generate markdown summary
        markdown_summary = self._generate_markdown_summary(structured_output)

        return {
            "success": True,
            "structured_output": structured_output,
            "markdown_summary": markdown_summary,
            "export_ready": True,
            "format_quality": "professional"
        }

    async def _coordination_action(self, state: EnhancedHRAgentState, task: str) -> Dict[str, Any]:
        """Coordination and workflow management"""

        return {
            "success": True,
            "coordination_type": "workflow_management",
            "task_completed": task,
            "next_steps": ["proceed_to_next_agent"],
            "coordination_quality": "standard"
        }

    # Helper methods (same as before but simplified)
    def _generate_market_insights(self, research_results: Dict) -> Dict[str, Any]:
        insights = {
            "market_temperature": "moderate",
            "hiring_difficulty": "medium",
            "key_trends": []
        }

        hot_roles = sum(1 for data in research_results.values()
                       if data.get("market_insights", {}).get("market_temperature") == "hot")

        if hot_roles / max(len(research_results), 1) > 0.5:
            insights["market_temperature"] = "hot"

        return insights

    async def _generate_enhanced_job_description(self, role: EnhancedRoleRequirement, company_context: Dict, market_data: Dict) -> str:
        """Generate enhanced job description using AI"""

        prompt = f"""
        Create a compelling job description for a {role.role_name} position.

        Company: {company_context.get('company_name', 'Our Company')}
        Industry: {company_context.get('industry', 'Technology')}
        Stage: {company_context.get('stage', 'Growth')}

        Role: {role.role_name}
        Skills: {', '.join(role.skills_required)}

        Create sections for: company intro, role overview, responsibilities, requirements, benefits.
        Length: 300-500 words, professional but engaging tone.
        """

        try:
            openai_client = self.tools.get("openai_client")
            if openai_client:
                response = await openai_client.ainvoke([
                    SystemMessage(content="You are an expert job description writer."),
                    HumanMessage(content=prompt)
                ])
                return response.content
        except Exception:
            pass

        # Fallback job description
        return f"""
# {role.role_name} - {company_context.get('company_name', 'Our Company')}

## About Us
Join {company_context.get('company_name', 'our company')}, a {company_context.get('stage', 'growing')} {company_context.get('industry', 'technology')} company.

## The Role
We're seeking a talented {role.role_name} to join our team. This is an opportunity to work on cutting-edge projects.

## Key Responsibilities
- Develop solutions using {', '.join(role.skills_required[:3])}
- Collaborate with cross-functional teams
- Contribute to technical decisions

## Requirements
- Experience with {', '.join(role.skills_required)}
- Strong problem-solving skills
- Passion for {role.role_category}

## What We Offer
- Competitive salary and equity
- Flexible work arrangements
- Growth opportunities
"""

    def _generate_integration_recommendations(self, roles: List[EnhancedRoleRequirement], company_context: Dict, research_data: Dict) -> List[str]:
        recommendations = [
            "Set up modern ATS with candidate scoring",
            "Implement structured interview processes",
            "Create candidate experience tracking"
        ]

        ai_roles = [r for r in roles if r.role_category == "ai_ml"]
        if ai_roles:
            recommendations.append("Partner with AI/ML communities for sourcing")

        return recommendations

    def _analyze_conversation_changes(self, state: EnhancedHRAgentState) -> Dict[str, Any]:
        return {
            "changes": [],
            "change_type": "incremental",
            "requires_full_reprocess": False
        }

    def _extract_user_preferences(self, state: EnhancedHRAgentState) -> Dict[str, Any]:
        preferences = {}
        company_context = state.get("company_context", {})
        if company_context:
            preferences["preferred_industry"] = company_context.get("industry")
            preferences["company_stage"] = company_context.get("stage")
        return preferences

    def _generate_markdown_summary(self, structured_output: Dict) -> str:
        if structured_output.get("type") == "conversation":
            return f"# Conversation Response\n\n{structured_output.get('response', '')}"

        hiring_plan = structured_output.get("hiring_plan", {})
        roles = hiring_plan.get("roles", [])

        markdown = f"""# Hiring Plan Summary
Generated: {structured_output.get('generated_at', 'Unknown')}

## Roles Overview ({len(roles)})
"""
        for role in roles:
            markdown += f"- **{role.get('role_name', 'Unknown')}** ({role.get('role_category', 'general')})\n"

        return markdown

    async def _reflection_and_learning(self, reasoning: Dict, action_result: Dict, state: EnhancedHRAgentState) -> Dict[str, Any]:
        """Enhanced reflection and learning"""

        success = action_result.get("success", False)
        quality_score = 0.8 if success else 0.3

        return {
            "success_assessment": success,
            "quality_score": quality_score,
            "agent_performance": "good" if success else "needs_improvement",
            "lessons_learned": [f"{self.role.value} agent completed task successfully" if success else f"{self.role.value} agent encountered issues"],
            "recommendations": ["Continue with next workflow step" if success else "Review and retry with adjusted approach"]
        }

class EnhancedHRAgentOrchestrator:
    """Enhanced multi-agent orchestrator with intent-based routing"""

    def __init__(self):
        self.tools = self._initialize_enhanced_tools()
        self.agents = self._initialize_enhanced_agents()
        self.intent_classifier = IntentClassifier(self.tools["openai_client"])
        self.conversational_agent = ConversationalAgent(self.tools["openai_client"])
        self.memory_saver = MemorySaver()
        self.graph = self._build_enhanced_graph()
        self.session_analytics = {"total_sessions": 0, "successful_sessions": 0, "conversation_sessions": 0}

    def _initialize_enhanced_tools(self) -> Dict[str, Any]:
        """Initialize enhanced tool suite"""

        openai_client = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4",
            temperature=0.7
        )

        tools = {
            "openai_client": openai_client,
            "tavily_search": TavilySearchTool(),
            "email_writer": EnhancedEmailWriterTool(openai_client),
            "checklist_builder": EnhancedChecklistBuilderTool(),
            "role_detector": AdvancedRoleDetector(nlp) if nlp else None
        }

        return tools

    def _initialize_enhanced_agents(self) -> Dict[AgentRole, EnhancedReactAgent]:
        """Initialize enhanced agent suite"""

        agents = {}
        for role in AgentRole:
            agents[role] = EnhancedReactAgent(role, self.tools)

        return agents

    def _build_enhanced_graph(self) -> StateGraph:
        """Build enhanced LangGraph workflow with intent routing"""

        workflow = StateGraph(EnhancedHRAgentState)

        # Add nodes with intent classification first
        workflow.add_node("intent_classifier", self._intent_classification_node)
        workflow.add_node("conversational_agent", self._conversational_agent_node)
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("nlp_analyzer", self._nlp_analyzer_node)
        workflow.add_node("memory_manager", self._memory_manager_node)
        workflow.add_node("requirements_check", self._requirements_check_node)
        workflow.add_node("research_agent", self._research_agent_node)
        workflow.add_node("content_agent", self._content_agent_node)
        workflow.add_node("integration_agent", self._integration_agent_node)
        workflow.add_node("output_formatter", self._output_formatter_node)
        workflow.add_node("final_synthesis", self._final_synthesis_node)

        # Start with intent classification
        workflow.set_entry_point("intent_classifier")

        # Route based on intent
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_by_intent,
            {
                "conversation": "conversational_agent",
                "hiring": "coordinator"
            }
        )

        # Conversational flow (simple, no tools)
        workflow.add_edge("conversational_agent", "output_formatter")

        # Hiring flow (complex, with all tools)
        workflow.add_edge("coordinator", "memory_manager")
        workflow.add_edge("memory_manager", "nlp_analyzer")
        workflow.add_conditional_edges(
            "nlp_analyzer",
            self._should_request_clarifications,
            {
                "needs_clarification": "requirements_check",
                "proceed": "research_agent"
            }
        )
        workflow.add_edge("requirements_check", END)
        workflow.add_edge("research_agent", "content_agent")
        workflow.add_edge("content_agent", "integration_agent")
        workflow.add_edge("integration_agent", "final_synthesis")

        # Both flows end at output formatter then END
        workflow.add_edge("output_formatter", END)
        workflow.add_edge("final_synthesis", END)

        return workflow.compile(checkpointer=self.memory_saver)

    async def _intent_classification_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Classify user intent first"""

        messages = state.get("messages", [])
        user_input = messages[-1].content if messages else ""

        intent_result = await self.intent_classifier.classify_intent(user_input)

        state["intent_classification"] = intent_result
        state["workflow_stage"] = "intent_classification"
        state["current_agent"] = "intent_classifier"

        print(f"🧠 Intent Classifier: {intent_result['intent']} (confidence: {intent_result['confidence']:.2f})")

        return state

    def _route_by_intent(self, state: EnhancedHRAgentState) -> Literal["conversation", "hiring"]:
        """Route based on classified intent"""

        intent = state.get("intent_classification", {}).get("intent", "hiring")
        return intent

    async def _conversational_agent_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Handle conversational queries without tools"""

        messages = state.get("messages", [])
        user_input = messages[-1].content if messages else ""

        # Generate conversational response
        response = await self.conversational_agent.generate_response(user_input)

        # Update state for conversational flow
        state["agent_outputs"]["conversation"] = {
            "response": response,
            "type": "conversational",
            "tools_used": [],
            "intent": state.get("intent_classification", {}).get("intent", "conversation")
        }

        state["workflow_stage"] = "conversation_complete"
        state["current_agent"] = "conversational"
        state["completion_percentage"] = 100

        # Update analytics
        self.session_analytics["conversation_sessions"] += 1

        print(f"💬 Conversational Agent: Response generated without tool calls")

        return state

    async def _coordinator_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Enhanced coordinator with session management"""

        coordinator = self.agents[AgentRole.COORDINATOR]
        result = await coordinator.process_with_enhanced_reasoning(state, "coordinate_session")

        # Initialize or update session
        if not state.get("session_id"):
            state["session_id"] = str(uuid.uuid4())
            self.session_analytics["total_sessions"] += 1

        state["workflow_stage"] = "coordination"
        state["current_agent"] = "coordinator"
        state["completion_percentage"] = 10

        print(f"🎯 Coordinator: Session {state['session_id'][:8]} initialized")

        return state

    async def _nlp_analyzer_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Enhanced NLP analysis with role detection"""

        nlp_agent = self.agents[AgentRole.NLP_ANALYZER]
        result = await nlp_agent.process_with_enhanced_reasoning(state, "analyze_roles")

        # Update state with analysis results
        action_result = result["action"]
        state["detected_roles"] = action_result.get("detected_roles", [])
        state["role_analysis_confidence"] = action_result.get("confidence", 0.5)
        state["clarifications_needed"] = action_result.get("clarifications_needed", [])
        state["agent_outputs"]["nlp_analysis"] = action_result

        state["workflow_stage"] = "nlp_analysis"
        state["current_agent"] = "nlp_analyzer"
        state["completion_percentage"] = 25

        print(f"🧠 NLP Analyzer: Detected {len(state['detected_roles'])} roles with {state['role_analysis_confidence']:.2f} confidence")

        return state

    async def _memory_manager_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Enhanced memory management"""

        memory_agent = self.agents[AgentRole.MEMORY_MANAGER]
        result = await memory_agent.process_with_enhanced_reasoning(state, "manage_memory")

        # Update conversation memory
        action_result = result["action"]
        state["conversation_memory"] = action_result.get("memory_update", {})
        state["is_followup"] = action_result.get("is_followup", False)
        state["agent_outputs"]["memory"] = action_result

        state["workflow_stage"] = "memory_management"
        state["current_agent"] = "memory_manager"

        print(f"🧠 Memory Manager: Session continuity managed, followup: {state['is_followup']}")

        return state

    async def _requirements_check_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Requirements clarification checkpoint"""

        state["workflow_stage"] = "clarification_needed"
        state["completion_percentage"] = 30

        print(f"❓ Requirements Check: {len(state.get('clarifications_needed', []))} clarifications needed")

        return state

    def _should_request_clarifications(self, state: EnhancedHRAgentState) -> Literal["needs_clarification", "proceed"]:
        """Decide whether to request clarifications or proceed"""

        clarifications = state.get("clarifications_needed", [])
        confidence = state.get("role_analysis_confidence", 0.0)

        # Request clarifications if confidence is low or specific info is missing
        if len(clarifications) > 0 and confidence < 0.8:
            return "needs_clarification"
        else:
            return "proceed"

    async def _research_agent_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Enhanced research with Tavily"""

        research_agent = self.agents[AgentRole.RESEARCH]
        result = await research_agent.process_with_enhanced_reasoning(state, "market_research")

        action_result = result["action"]
        state["agent_outputs"]["research"] = action_result
        state["industry_insights"] = action_result.get("market_insights", {})

        state["workflow_stage"] = "research"
        state["current_agent"] = "research"
        state["completion_percentage"] = 50

        print(f"🔍 Research Agent: Market data gathered for {action_result.get('roles_researched', 0)} roles")

        return state

    async def _content_agent_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Enhanced content generation"""

        content_agent = self.agents[AgentRole.CONTENT]
        result = await content_agent.process_with_enhanced_reasoning(state, "generate_content")

        action_result = result["action"]
        state["agent_outputs"]["content"] = action_result

        state["workflow_stage"] = "content_generation"
        state["current_agent"] = "content"
        state["completion_percentage"] = 70

        print(f"📝 Content Agent: Generated content for {action_result.get('roles_processed', 0)} roles")

        return state

    async def _integration_agent_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Enhanced integration planning"""

        integration_agent = self.agents[AgentRole.INTEGRATION]
        result = await integration_agent.process_with_enhanced_reasoning(state, "plan_integration")

        action_result = result["action"]
        state["agent_outputs"]["integration"] = action_result

        state["workflow_stage"] = "integration"
        state["current_agent"] = "integration"
        state["completion_percentage"] = 85

        complexity = action_result.get("complexity_score", 0.6)
        print(f"⚙️ Integration Agent: Workflow planned with {complexity:.2f} complexity score")

        return state

    async def _output_formatter_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Enhanced output formatting for both conversation and hiring"""

        formatter_agent = self.agents[AgentRole.OUTPUT_FORMATTER]
        result = await formatter_agent.process_with_enhanced_reasoning(state, "format_output")

        action_result = result["action"]
        state["structured_output"] = action_result.get("structured_output", {})
        state["export_formats"] = action_result.get("export_formats", [])
        state["agent_outputs"]["formatter"] = action_result

        intent = state.get("intent_classification", {}).get("intent", "hiring")
        state["workflow_stage"] = "formatting_complete"
        state["current_agent"] = "formatter"

        print(f"📊 Output Formatter: {intent} response formatted")

        return state

    async def _final_synthesis_node(self, state: EnhancedHRAgentState) -> EnhancedHRAgentState:
        """Final synthesis and completion"""

        # Mark as successful completion
        self.session_analytics["successful_sessions"] += 1

        # Generate final summary
        state["workflow_stage"] = "complete"
        state["completion_percentage"] = 100
        state["session_analytics"] = {
            "total_agents_used": len(state.get("agent_outputs", {})),
            "workflow_efficiency": state["completion_percentage"] / 100,
            "session_success": True,
            "completion_time": datetime.now().isoformat()
        }

        print(f"🎉 Final Synthesis: Session completed successfully with {len(state.get('agent_outputs', {}))} agents")

        return state

    async def process_hiring_request(
        self,
        user_input: str,
        company_context: Dict = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """Main processing function with enhanced workflow and intent routing"""

        # Initialize enhanced state
        config = {"configurable": {"thread_id": session_id or str(uuid.uuid4())}}

        initial_state = EnhancedHRAgentState(
            messages=[HumanMessage(content=user_input)],
            session_id=session_id or "",
            conversation_memory={},
            detected_roles=[],
            role_analysis_confidence=0.0,
            clarifications_needed=[],
            company_context=company_context or {},
            industry_insights={},
            current_agent="",
            agent_outputs={},
            workflow_stage="initialization",
            completion_percentage=0,
            is_followup=bool(session_id),
            change_detection={},
            incremental_updates=[],
            structured_output={},
            export_formats=[],
            session_analytics={}
        )

        try:
            print(f"🚀 Starting enhanced HR agent workflow with intent classification...")

            # Run the enhanced graph
            final_state = await self.graph.ainvoke(initial_state, config)

            return final_state

        except Exception as e:
            print(f"❌ Workflow error: {e}")

            # Return error state
            error_state = initial_state.copy()
            error_state.update({
                "workflow_stage": "error",
                "error": str(e),
                "completion_percentage": 0
            })

            return error_state

    def get_enhanced_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics"""

        success_rate = 0
        if self.session_analytics["total_sessions"] > 0:
            success_rate = self.session_analytics["successful_sessions"] / self.session_analytics["total_sessions"]

        return {
            "session_analytics": self.session_analytics,
            "success_rate": success_rate,
            "conversation_rate": self.session_analytics["conversation_sessions"] / max(self.session_analytics["total_sessions"], 1),
            "agent_performance": {
                agent_role.value: len(agent.performance_history)
                for agent_role, agent in self.agents.items()
            },
            "tool_usage": {
                "tavily_search": len(self.tools["tavily_search"].cache),
                "email_templates": len(self.tools["email_writer"].email_cache),
                "checklists": len(self.tools["checklist_builder"].checklist_cache)
            }
        }

print("✅ Enhanced multi-agent system with intent classification and conditional routing complete!")

# ==================== CELL SEPARATOR ====================

def visualize_langgraph_workflow():
    """Visualize the LangGraph workflow structure"""

    try:
        # Try to use built-in LangGraph visualization if available
        import matplotlib.pyplot as plt
        import networkx as nx

        # Create a directed graph representation
        G = nx.DiGraph()

        # Add nodes
        nodes = [
            ("intent_classifier", "🧠 Intent\nClassifier"),
            ("conversational_agent", "💬 Conversational\nAgent"),
            ("coordinator", "🎯 Coordinator"),
            ("memory_manager", "💾 Memory\nManager"),
            ("nlp_analyzer", "🔍 NLP\nAnalyzer"),
            ("requirements_check", "❓ Requirements\nCheck"),
            ("research_agent", "📊 Research\nAgent"),
            ("content_agent", "📝 Content\nAgent"),
            ("integration_agent", "⚙️ Integration\nAgent"),
            ("output_formatter", "📋 Output\nFormatter"),
            ("final_synthesis", "🎉 Final\nSynthesis"),
            ("END", "🏁 END")
        ]

        for node_id, label in nodes:
            G.add_node(node_id, label=label)

        # Add edges
        edges = [
            ("intent_classifier", "conversational_agent", "conversation"),
            ("intent_classifier", "coordinator", "hiring"),
            ("conversational_agent", "output_formatter", ""),
            ("coordinator", "memory_manager", ""),
            ("memory_manager", "nlp_analyzer", ""),
            ("nlp_analyzer", "requirements_check", "needs_clarification"),
            ("nlp_analyzer", "research_agent", "proceed"),
            ("requirements_check", "END", ""),
            ("research_agent", "content_agent", ""),
            ("content_agent", "integration_agent", ""),
            ("integration_agent", "final_synthesis", ""),
            ("output_formatter", "END", ""),
            ("final_synthesis", "END", "")
        ]

        for source, target, condition in edges:
            G.add_edge(source, target, condition=condition)

        # Create visualization
        plt.figure(figsize=(16, 12))

        # Define positions for better layout
        pos = {
            "intent_classifier": (0, 0),
            "conversational_agent": (-2, -1),
            "coordinator": (2, -1),
            "memory_manager": (2, -2),
            "nlp_analyzer": (2, -3),
            "requirements_check": (0, -4),
            "research_agent": (4, -4),
            "content_agent": (4, -5),
            "integration_agent": (4, -6),
            "final_synthesis": (4, -7),
            "output_formatter": (1, -8),
            "END": (1, -9)
        }

        # Draw nodes with different colors for different types
        node_colors = {
            "intent_classifier": "lightblue",
            "conversational_agent": "lightgreen",
            "coordinator": "lightcoral",
            "memory_manager": "lightyellow",
            "nlp_analyzer": "lightpink",
            "requirements_check": "orange",
            "research_agent": "lightsteelblue",
            "content_agent": "lightseagreen",
            "integration_agent": "plum",
            "final_synthesis": "gold",
            "output_formatter": "lightgray",
            "END": "red"
        }

        for node in G.nodes():
            nx.draw_networkx_nodes(G, pos, nodelist=[node],
                                 node_color=node_colors.get(node, "lightblue"),
                                 node_size=2000)

        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

        # Add labels
        labels = {node: G.nodes[node]["label"] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        # Add edge labels for conditions
        edge_labels = {}
        for source, target, data in G.edges(data=True):
            if data.get("condition"):
                edge_labels[(source, target)] = data["condition"]

        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

        plt.title("Enhanced HR Agent v2.0 - LangGraph Workflow", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        # Save the plot
        plt.savefig('hr_agent_workflow.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ LangGraph workflow visualization created!")

        return G

    except ImportError:
        print("⚠️ Matplotlib/NetworkX not available. Here's the text representation:")

        workflow_text = """
🎯 ENHANCED HR AGENT v2.0 - LANGGRAPH WORKFLOW

START → Intent Classifier
         ├─ [conversation] → Conversational Agent → Output Formatter → END
         └─ [hiring] → Coordinator → Memory Manager → NLP Analyzer
                                                           ├─ [needs_clarification] → Requirements Check → END
                                                           └─ [proceed] → Research Agent → Content Agent → Integration Agent → Final Synthesis → END

🔧 WORKFLOW EXPLANATION:

1. **Intent Classifier** 🧠
   - Determines if input is conversational or hiring-related
   - Routes to appropriate flow

2. **Conversational Flow** 💬 (for greetings, help, general questions)
   - Conversational Agent → Output Formatter → END
   - NO TOOLS CALLED - efficient and fast

3. **Hiring Flow** 🎯 (for actual hiring requests)
   - Full multi-agent coordination with tool integration
   - Conditional routing based on completeness
   - All three core tools utilized

🎪 KEY FEATURES:
✅ Conditional routing prevents unnecessary tool calls
✅ Early exit for conversational queries
✅ Memory persistence across sessions
✅ Adaptive workflow based on role complexity
✅ Professional output formatting for both flows
"""

        print(workflow_text)
        return None

def display_current_state(state: EnhancedHRAgentState):
    """Display current workflow state for debugging"""

    print("🔍 CURRENT WORKFLOW STATE")
    print("=" * 50)

    # Basic state info
    print(f"Session ID: {state.get('session_id', 'Unknown')}")
    print(f"Workflow Stage: {state.get('workflow_stage', 'Unknown')}")
    print(f"Current Agent: {state.get('current_agent', 'Unknown')}")
    print(f"Completion: {state.get('completion_percentage', 0)}%")

    # Intent classification
    intent_data = state.get("intent_classification", {})
    if intent_data:
        print(f"\n🧠 Intent Analysis:")
        print(f"  Intent: {intent_data.get('intent', 'Unknown')}")
        print(f"  Confidence: {intent_data.get('confidence', 0):.2f}")
        print(f"  Requires Tools: {intent_data.get('requires_tools', False)}")
        print(f"  Reasoning: {intent_data.get('reasoning', 'N/A')}")

    # Detected roles (if any)
    detected_roles = state.get("detected_roles", [])
    if detected_roles:
        print(f"\n💼 Detected Roles ({len(detected_roles)}):")
        for i, role in enumerate(detected_roles, 1):
            print(f"  {i}. {role.get('role_name', 'Unknown')} (confidence: {role.get('confidence_score', 0):.2f})")

    # Agent outputs
    agent_outputs = state.get("agent_outputs", {})
    if agent_outputs:
        print(f"\n🤖 Agent Outputs ({len(agent_outputs)}):")
        for agent, output in agent_outputs.items():
            if isinstance(output, dict):
                print(f"  {agent}: {output.get('type', 'processed')}")
            else:
                print(f"  {agent}: completed")

    # Memory info
    conversation_memory = state.get("conversation_memory", {})
    if conversation_memory:
        print(f"\n💾 Memory Status:")
        print(f"  Session Count: {conversation_memory.get('conversation_count', 0)}")
        print(f"  Is Followup: {state.get('is_followup', False)}")

    print("=" * 50)

print("✅ LangGraph visualization and state management tools ready!")

# ==================== CELL SEPARATOR ====================

# ========================================
# REPLACE CELL 5: Updated Gradio Interface with Conditional Response Handling
# ========================================

# Global orchestrator instance with updated logic
hr_orchestrator = EnhancedHRAgentOrchestrator()

# Session management
active_sessions = {}

# ========== EXPORT FUNCTIONS ==========
def export_results_json(session_id: str) -> str:
    """Export session results as JSON"""
    if session_id not in active_sessions:
        return json.dumps({"error": "Session not found"}, indent=2)

    session_data = active_sessions[session_id]

    # Build clean export structure
    export_data = {
        "session_id": session_id,
        "generated_at": datetime.now().isoformat(),
        "intent": session_data.get("intent_classification", {}),
        "company_context": session_data.get("company_context", {}),
        "detected_roles": session_data.get("detected_roles", []),
        "workflow_stage": session_data.get("workflow_stage", ""),
        "completion_percentage": session_data.get("completion_percentage", 0),
        "results": {
            "market_research": session_data.get("agent_outputs", {}).get("research", {}).get("research_data", {}),
            "job_descriptions": session_data.get("agent_outputs", {}).get("content", {}).get("content_deliverables", {}).get("job_descriptions", {}),
            "email_templates": session_data.get("agent_outputs", {}).get("content", {}).get("content_deliverables", {}).get("email_templates", {}),
            "hiring_workflow": session_data.get("agent_outputs", {}).get("integration", {}).get("hiring_checklist", {})
        },
        "analytics": session_data.get("session_analytics", {})
    }

    return json.dumps(export_data, indent=2, default=str)

def export_results_markdown(session_id: str) -> str:
    """Export session results as Markdown report"""
    if session_id not in active_sessions:
        return "# Error\nSession not found"

    session_data = active_sessions[session_id]
    detected_roles = session_data.get("detected_roles", [])
    company_context = session_data.get("company_context", {})
    agent_outputs = session_data.get("agent_outputs", {})

    # Build markdown report
    md_content = f"""# Hiring Plan Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Session ID:** {session_id}
**Company:** {company_context.get('company_name', 'Unknown')}

## Executive Summary
- **Roles to Fill:** {len(detected_roles)}
- **Completion Status:** {session_data.get('completion_percentage', 0)}%
- **Workflow Stage:** {session_data.get('workflow_stage', 'Unknown').replace('_', ' ').title()}

## Roles Overview
"""

    # Add role details
    for role in detected_roles:
        md_content += f"""
### {role.get('role_name', 'Unknown Role')}
- **Category:** {role.get('role_category', 'Unknown')}
- **Technical Level:** {role.get('technical_level', 'Unknown')}
- **Confidence Score:** {role.get('confidence_score', 0):.2f}
"""
        if role.get('skills_required'):
            md_content += f"- **Key Skills:** {', '.join(role['skills_required'][:5])}\n"

    # Add market research if available
    research_data = agent_outputs.get("research", {}).get("research_data", {})
    if research_data:
        md_content += "\n## Market Research\n"
        for role_name, data in research_data.items():
            salary_data = data.get("salary_data", {}).get("base", {})
            insights = data.get("market_insights", {})

            md_content += f"""
### {role_name}
- **Salary Range:** ${salary_data.get('min', 0):,} - ${salary_data.get('max', 0):,}
- **Market Median:** ${salary_data.get('median', 0):,}
- **Market Temperature:** {insights.get('market_temperature', 'Unknown').title()}
"""

    # Add workflow timeline if available
    integration_data = agent_outputs.get("integration", {})
    workflow_data = integration_data.get("hiring_checklist", {}) if integration_data else {}

    if workflow_data and workflow_data.get("phases"):
        md_content += f"""
## Hiring Workflow
- **Total Timeline:** {workflow_data.get('adaptive_timeline', 'TBD')} days
- **Number of Phases:** {len(workflow_data.get('phases', []))}

### Timeline Breakdown
"""
        for phase in workflow_data.get("phases", [])[:5]:
            md_content += f"- **{phase.get('phase', 'Unknown')}:** {phase.get('duration_days', 'TBD')} days\n"

    # Add job descriptions preview if available
    content_data = agent_outputs.get("content", {}).get("content_deliverables", {})
    job_descriptions = content_data.get("job_descriptions", {}) if content_data else {}

    if job_descriptions:
        md_content += "\n## Job Descriptions Generated\n"
        for role_name in job_descriptions.keys():
            md_content += f"- ✅ {role_name}\n"

    # Add next steps
    md_content += """
## Next Steps
1. Review and approve job descriptions
2. Set up hiring pipeline according to workflow
3. Begin candidate sourcing
4. Execute hiring plan with regular check-ins

---
*Generated by Enhanced HR Agent v2.0*
"""

    return md_content

async def process_hiring_request_enhanced_ui(
    user_message,
    company_name,
    industry,
    stage,
    team_size,
    budget,
    session_id=None
):
    """Enhanced UI processing with conditional response handling"""

    # Build company context
    company_context = {
        "company_name": company_name or "Our Company",
        "industry": industry or "Technology",
        "stage": stage or "Series A",
        "team_size": int(team_size) if team_size else 20,
        "budget_range": budget,
        "location": "United States"
    }

    try:
        print(f"🎯 Processing: {user_message[:50]}...")

        # Process with enhanced orchestrator
        result = await hr_orchestrator.process_hiring_request(
            user_message,
            company_context,
            session_id
        )

        # Store session for follow-ups
        result_session_id = result.get("session_id", session_id or str(uuid.uuid4()))
        active_sessions[result_session_id] = result

        # Format response based on intent and workflow stage
        if result.get("error"):
            return format_error_response(result), result_session_id

        # Check intent and format accordingly
        intent_data = result.get("intent_classification", {})
        intent = intent_data.get("intent", "hiring")

        if intent == "conversation":
            return format_conversational_response(result), result_session_id
        elif result.get("workflow_stage") == "clarification_needed":
            return format_clarification_response(result), result_session_id
        else:
            return format_success_response(result), result_session_id

    except Exception as e:
        error_msg = f"❌ Processing Error: {str(e)}"
        return (error_msg, "", "", "", ""), session_id

def format_conversational_response(result):
    """Format conversational (non-hiring) response"""

    conversation_data = result.get("agent_outputs", {}).get("conversation", {})
    response_text = conversation_data.get("response", "")
    intent_data = result.get("intent_classification", {})

    # Status showing this was a conversational query
    status = f"💬 **Conversational Response**\n\n"
    status += f"**Intent:** {intent_data.get('intent', 'conversation').title()}\n"
    status += f"**Confidence:** {intent_data.get('confidence', 0):.2f}\n"
    status += f"**Tools Used:** None (efficient conversational mode)\n"
    status += f"**Session ID:** {result.get('session_id', 'Unknown')[:8]}...\n"

    # Main response in the roles section for visibility
    main_response = f"🤖 **AI Assistant Response:**\n\n{response_text}"

    # Clear other sections since no hiring analysis was performed
    empty_section = "ℹ️ *No analysis performed - this was a conversational query*"

    return (status, main_response, empty_section, empty_section, empty_section)

def format_error_response(result):
    """Format error response for UI"""
    error_msg = f"❌ **Error:** {result.get('error', 'Unknown error')}"
    return (error_msg, "", "", "", "")

def format_clarification_response(result):
    """Format clarification request for UI"""
    clarifications = result.get("clarifications_needed", [])
    detected_roles = result.get("detected_roles", [])
    intent_data = result.get("intent_classification", {})

    status = f"🤖 **Initial Analysis Complete**\n\n"
    status += f"**Intent:** {intent_data.get('intent', 'hiring').title()}\n"
    status += f"**Detected Roles:** {len(detected_roles)}\n"
    for role in detected_roles:
        status += f"- {role.get('role_name', 'Unknown')} ({role.get('confidence_score', 0):.2f} confidence)\n"

    clarifications_text = "❓ **Please provide additional information:**\n\n"
    for i, clarification in enumerate(clarifications, 1):
        clarifications_text += f"{i}. {clarification}\n"

    clarifications_text += "\n💡 **Tip:** The more details you provide, the better our recommendations will be!"

    return (status, clarifications_text, "", "", "")

def format_success_response(result):
    """Format successful hiring analysis response"""

    # Extract key data
    detected_roles = result.get("detected_roles", [])
    agent_outputs = result.get("agent_outputs", {})
    completion = result.get("completion_percentage", 0)
    intent_data = result.get("intent_classification", {})

    # Status summary with intent info
    status = f"✅ **Analysis Complete ({completion}%)**\n\n"
    status += f"**Intent:** {intent_data.get('intent', 'hiring').title()}\n"
    status += f"**Confidence:** {intent_data.get('confidence', 0):.2f}\n"
    status += f"**Session ID:** {result.get('session_id', 'Unknown')[:8]}...\n"
    status += f"**Roles Analyzed:** {len(detected_roles)}\n"
    status += f"**Agents Used:** {len(agent_outputs)}\n"
    status += f"**Tools Called:** {intent_data.get('requires_tools', True)}\n"

    # Roles breakdown
    roles_text = "💼 **Detected Roles:**\n\n"
    for role in detected_roles:
        roles_text += f"**{role.get('role_name', 'Unknown Role')}**\n"
        roles_text += f"- Category: {role.get('role_category', 'general')}\n"
        roles_text += f"- Level: {role.get('technical_level', 'mid')}\n"
        roles_text += f"- Confidence: {role.get('confidence_score', 0):.2f}\n"
        if role.get('skills_required'):
            roles_text += f"- Key Skills: {', '.join(role['skills_required'][:3])}\n"
        roles_text += "\n"

    # Market research (only if research agent ran)
    research_data = agent_outputs.get("research", {}).get("research_data", {})
    market_text = "💰 **Market Research:**\n\n"

    if research_data:
        for role_name, data in research_data.items():
            market_text += f"**{role_name}:**\n"
            salary_data = data.get("salary_data", {}).get("base", {})

            if salary_data.get("hourly"):
                market_text += f"- Hourly Rate: ${salary_data.get('median', 25)}/hour\n"
                market_text += f"- Annual Equivalent: ~${salary_data.get('median', 25) * 2080:,.0f}\n"
            else:
                market_text += f"- Salary Range: ${salary_data.get('min', 80000):,} - ${salary_data.get('max', 150000):,}\n"
                market_text += f"- Market Median: ${salary_data.get('median', 115000):,}\n"

            # Add market insights
            insights = data.get("market_insights", {})
            if insights:
                market_text += f"- Market Temperature: {insights.get('market_temperature', 'moderate').title()}\n"
                market_text += f"- Hiring Difficulty: {insights.get('hiring_difficulty', 'medium').title()}\n"

            market_text += "\n"
    else:
        market_text += "ℹ️ *Research data not generated for this query type*\n"

    # Workflow timeline (only if integration agent ran)
    integration_data = agent_outputs.get("integration", {})
    workflow_data = integration_data.get("hiring_checklist", {}) if integration_data else {}

    workflow_text = "📋 **Hiring Workflow:**\n\n"
    if workflow_data:
        timeline = workflow_data.get("adaptive_timeline", 30)
        phases = workflow_data.get("phases", [])

        workflow_text += f"**Timeline:** {timeline} days\n"
        workflow_text += f"**Phases:** {len(phases)}\n\n"

        # Show first few phases
        for i, phase in enumerate(phases[:4], 1):
            workflow_text += f"**{i}. {phase.get('phase', 'Unknown Phase')}**\n"
            workflow_text += f"Duration: {phase.get('duration_days', 'TBD')} days\n"
            workflow_text += f"Owner: {phase.get('owner', 'TBD')}\n\n"

        if len(phases) > 4:
            workflow_text += f"... and {len(phases) - 4} more phases\n\n"
    else:
        workflow_text += "ℹ️ *Workflow not generated for this query type*\n"

    # Job descriptions and content (only if content agent ran)
    content_data = agent_outputs.get("content", {}).get("content_deliverables", {})
    job_descriptions = content_data.get("job_descriptions", {}) if content_data else {}

    content_text = "📝 **Generated Content:**\n\n"

    if job_descriptions:
        for role_name, jd in job_descriptions.items():
            content_text += f"**{role_name} Job Description:**\n"
            # Show preview of job description
            if isinstance(jd, str) and len(jd) > 200:
                content_text += f"{jd[:200]}...\n\n"
            elif isinstance(jd, str):
                content_text += f"{jd}\n\n"
            else:
                content_text += "Job description generated successfully.\n\n"

        # Email templates
        email_templates = content_data.get("email_templates", {})
        if email_templates:
            content_text += "**📧 Email Templates Generated:**\n"
            for role_name, templates in email_templates.items():
                content_text += f"- {role_name}: Outreach & Interview emails\n"
            content_text += "\n"
    else:
        content_text += "ℹ️ *Content not generated for this query type*\n"

    return (status, roles_text, market_text, workflow_text, content_text)

def sync_process_enhanced_ui(user_message, company_name, industry, stage, team_size, budget, session_id=None):
    """Synchronous wrapper for enhanced UI processing"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        process_hiring_request_enhanced_ui(user_message, company_name, industry, stage, team_size, budget, session_id)
    )

