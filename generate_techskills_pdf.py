from fpdf import FPDF


class TechSkillsPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 20)
        self.cell(0, 12, "Adeesha Perera - Technical Skills", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 8, "Machine Learning Engineer | Data Scientist", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(4)
        self.set_draw_color(52, 73, 94)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(41, 128, 185)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(41, 128, 185)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def skill_row(self, category, skills, project_ref=None):
        self.set_font("Helvetica", "B", 10)
        self.cell(50, 6, category, new_x="END")
        self.set_font("Helvetica", "", 9)
        if project_ref:
            self.set_text_color(80, 80, 80)
            self.cell(0, 6, skills, new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(41, 128, 185)
            self.cell(50, 5, "", new_x="END")
            self.cell(0, 5, project_ref, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
        else:
            self.cell(0, 6, skills, new_x="LMARGIN", new_y="NEXT")

    def project_entry(self, name, tech, details):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, name, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Technologies: {tech}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, details)
        self.ln(2)


pdf = TechSkillsPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# ML & Data Science
pdf.section_title("Machine Learning & Data Science")
pdf.skill_row("Frameworks:", "PyTorch, TensorFlow, Scikit-learn, XGBoost, CatBoost, Hugging Face Transformers")
pdf.skill_row("MLOps:", "MLflow, DAGsHub, Apache Airflow, Feast Feature Store, GitHub Actions CI/CD",
              "Used in: Real-Time Fraud Detection System")
pdf.skill_row("Techniques:", "Supervised Learning, Feature Engineering, Model Optimization, A/B Testing")
pdf.skill_row("Evaluation:", "ROC-AUC, PR-AUC, F1-Score, Cross-Validation, RAGAS (RAG Evaluation)",
              "Used in: Hybrid RAG Portfolio Assistant")
pdf.skill_row("Streaming:", "Apache Kafka, Real-time Feature Computation",
              "Used in: Real-Time Fraud Detection System")
pdf.skill_row("Monitoring:", "Grafana, Prometheus, Evidently AI (Drift Detection)",
              "Used in: Real-Time Fraud Detection System")
pdf.ln(3)

# NLP & LLMs
pdf.section_title("NLP & Large Language Models")
pdf.skill_row("LLMs:", "GPT-OSS-120B, Llama 3.3, Groq Inference, Gemini Embeddings",
              "Used in: Hybrid RAG Portfolio Assistant")
pdf.skill_row("RAG:", "Hybrid Search (BM25 + k-NN), Cohere Reranking, Parent-Child Chunking",
              "Used in: Hybrid RAG Portfolio Assistant")
pdf.skill_row("Document Extraction:", "Docling (PDF/DOCX/PPTX/HTML), Layout Analysis, OCR",
              "Used in: Hybrid RAG Portfolio Assistant")
pdf.skill_row("Frameworks:", "LangChain, Langfuse Observability, Custom Guardrails",
              "Used in: Hybrid RAG Portfolio Assistant")
pdf.skill_row("Vector DBs:", "OpenSearch k-NN (HNSW, cosine similarity), Gemini Embedding-2 (768-dim)",
              "Used in: Hybrid RAG Portfolio Assistant")
pdf.skill_row("Evaluation:", "RAGAS (Faithfulness, Relevancy, Precision, Recall)",
              "Used in: Hybrid RAG Portfolio Assistant")
pdf.ln(3)

# Backend & API
pdf.section_title("Backend & API Development")
pdf.skill_row("Frameworks:", "FastAPI, Spring Boot (Java 17), Flask")
pdf.skill_row("Auth:", "JWT, OAuth 2.0, Spring Security",
              "Used in: Smart Campus Operations Hub")
pdf.skill_row("Databases:", "PostgreSQL, MongoDB, Redis, MySQL")
pdf.skill_row("API Design:", "REST APIs, WebSocket (Real-time), Webhook Integration",
              "Used in: Fraud Detection (WebSockets), RAG (Webhooks)")
pdf.ln(3)

# Frontend
pdf.section_title("Frontend Development")
pdf.skill_row("Frameworks:", "React, Next.js 16, TypeScript, Tailwind CSS v4")
pdf.skill_row("Animation:", "Framer Motion")
pdf.skill_row("State:", "React Hooks, Context API")
pdf.ln(3)

# Programming & Tools
pdf.section_title("Programming & Development Tools")
pdf.skill_row("Languages:", "Python, SQL, JavaScript/TypeScript, Java, Kotlin, R, PHP, Bash")
pdf.skill_row("Data:", "Pandas, NumPy, Spark, Hadoop, SQL, MongoDB")
pdf.skill_row("DevOps:", "Docker, Docker Compose, Kubernetes, AWS (EC2, S3, Lambda), Terraform")
pdf.skill_row("Version Control:", "Git, GitHub, GitLab, CI/CD Pipelines")
pdf.skill_row("Testing:", "Pytest, Evidently AI, Pandera")
pdf.ln(3)

# Cloud & Infrastructure
pdf.section_title("Cloud & Infrastructure")
pdf.skill_row("Cloud:", "AWS (Certified), GCP, Oracle Cloud Infrastructure")
pdf.skill_row("Containers:", "Docker, Docker Compose, Kubernetes, Coolify",
              "Used in: RAG (6 containers), Fraud Detection (Docker Compose)")
pdf.skill_row("Orchestration:", "Apache Airflow, GitHub Actions",
              "Used in: Fraud Detection (Automated MLOps)")
pdf.skill_row("Monitoring:", "Grafana, Prometheus, Langfuse, DAGsHub",
              "Used in: Fraud Detection (Grafana), RAG (Langfuse)")
pdf.ln(3)

# Data Engineering
pdf.section_title("Data Engineering & Analytics")
pdf.skill_row("ETL:", "Microsoft SSIS, Apache Airflow, Custom Pipelines")
pdf.skill_row("OLAP:", "SSIS, OLAP Cube, Power BI Dashboards",
              "Used in: ETL & Data Warehousing Project")
pdf.skill_row("Feature Store:", "Feast (Online + Offline)",
              "Used in: Real-Time Fraud Detection System")
pdf.skill_row("Message Queue:", "Apache Kafka",
              "Used in: Real-Time Fraud Detection System")
pdf.ln(3)

# Projects
pdf.add_page()
pdf.section_title("Featured Projects - Applied Skills")

pdf.project_entry(
    "Real-Time Fraud Detection System",
    "CatBoost, Feast, Kafka, Airflow, MLflow, Grafana, PostgreSQL, Redis, Docker",
    "Production fraud detection pipeline processing 10K transactions/sec with sub-15ms latency. "
    "ROC-AUC 0.9938, PR-AUC 0.94. 3-tier decision framework (Auto-Block, Manual Review, Soft Audit). "
    "Online/offline feature store with Feast, automated drift detection with Evidently AI, "
    "weekly retraining via Airflow DAGs, real-time dashboard with WebSockets."
)

pdf.project_entry(
    "Hybrid RAG Portfolio Assistant",
    "OpenSearch, Gemini, Cohere, Docling, LangChain, FastAPI, Docker, Coolify",
    "Production RAG system with hybrid search (BM25 + k-NN), parent-child chunking, "
    "Cohere rerank-v3.5 (Top 10 to Top 3), strategy router, input/output guardrails, "
    "portfolio-only classifier, conversation memory, LRU cache. "
    "Deployed on Oracle Cloud via Coolify with 6 containers. RAGAS evaluation."
)

pdf.project_entry(
    "Smart Campus Operations Hub",
    "Java 17, Spring Boot, Spring Security, MongoDB, React, OAuth 2.0",
    "Full-stack campus operations with JWT + OAuth 2.0 authentication, "
    "facilities management, bookings, maintenance tickets, and notifications."
)

pdf.project_entry(
    "EcoSprout - Carbon Credit Marketplace",
    "Next.js, TypeScript, Tailwind CSS, Clerk, Convex, Turbo",
    "Carbon credit marketplace connecting developers, verifiers, and buyers. "
    "Built educational hub with structured learning modules and community resources."
)

pdf.project_entry(
    "Movie Recommendation System",
    "Scikit-learn, Surprise, SBERT, Pandas, Jupyter",
    "Comprehensive recommendation engine with KNN, SVD, SBERT content-based, "
    "and hybrid ensemble models on MovieLens data."
)

output_path = "technical_skills.pdf"
pdf.output(output_path)
print(f"Created {output_path}")
