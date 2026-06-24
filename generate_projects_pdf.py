from fpdf import FPDF


class ProjectsPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 20)
        self.cell(0, 12, "Adeesha Perera - Projects", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 8, "Machine Learning Engineer | Data Scientist", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(4)
        self.set_draw_color(52, 73, 94)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def category_title(self, name, description):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(41, 128, 185)
        self.cell(0, 8, name, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, description, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(41, 128, 185)
        self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def project(self, title, status, description, tech, details=None):
        self.set_font("Helvetica", "B", 11)
        label = f"{title}"
        if status:
            label += f"  [{status}]"
        self.cell(0, 7, label, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5, description)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f"Technologies: {tech}", new_x="LMARGIN", new_y="NEXT")
        if details:
            self.set_font("Helvetica", "", 8)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 4, details)
        self.set_text_color(0, 0, 0)
        self.ln(3)


pdf = ProjectsPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Production & Portfolio
pdf.category_title("Production & Portfolio", "End-to-end systems built for real-world use")

pdf.project(
    "Real-Time Fraud Detection System",
    "LIVE",
    "Production-grade credit card fraud detection with CatBoost, Feast feature store, Kafka streaming, "
    "and automated MLOps lifecycle. Sub-15ms inference latency, ROC-AUC 0.9938, PR-AUC 0.94. "
    "3-tier decision framework: Auto-Block, Manual Review, Soft Audit Signal.",
    "CatBoost, Feast, Kafka, Airflow, Grafana, MLflow, DAGsHub, PostgreSQL, Redis, Docker",
    "Features: Online/offline feature store with zero training-serving skew, automated drift detection with "
    "Evidently AI, weekly retraining pipelines, real-time WebSocket dashboard, Prometheus + Grafana monitoring."
)

pdf.project(
    "Hybrid RAG Portfolio Assistant",
    "IN PROGRESS",
    "Production-grade Retrieval-Augmented Generation system with Docling document extraction, "
    "OpenSearch hybrid search (BM25 + k-NN), Cohere reranking, guardrails, and RAGAS evaluation. "
    "Deployed on Oracle Cloud via Coolify with webhook-triggered document processing.",
    "Python, FastAPI, LangChain, Groq, Gemini, Cohere, OpenSearch, MinIO, Docling, Langfuse, RAGAS, Docker",
    "Architecture: Parent-child chunking (2000/500 chars), strategy router (direct/rewrite/multi_query/step_back), "
    "input/output guardrails, portfolio-only classifier, conversation memory, LRU cache. "
    "6 containers: RAG API, Worker, OpenSearch, MinIO, Docling, OpenSearch Dashboards."
)

# University Projects
pdf.category_title("University Projects", "Academic coursework and research projects")

pdf.project(
    "Smart Campus Operations Hub",
    "",
    "Full-stack campus operations system for facilities, bookings, maintenance tickets, "
    "and notifications with JWT + OAuth 2.0 authentication.",
    "Java 17, Spring Boot, Spring Security, MongoDB, React, OAuth 2.0"
)

pdf.project(
    "EcoSprout - Carbon Credit Marketplace",
    "",
    "Carbon credit marketplace connecting project developers, verifiers, and buyers. "
    "Built the educational hub with structured learning modules and community resources.",
    "Next.js, TypeScript, Tailwind CSS, Clerk, Convex, Turbo"
)

pdf.project(
    "Wildlife Safari Project",
    "",
    "Full-stack wildlife safari management system with CRUD operations, built as a team project.",
    "PHP, JavaScript, MySQL, HTML/CSS"
)

pdf.project(
    "WanderGo",
    "",
    "Java web application built as a team project for university coursework.",
    "Java, JSP, Servlets, MySQL"
)

pdf.project(
    "Serenity Android App",
    "",
    "Native Android application built with Kotlin as a solo university project.",
    "Kotlin, Android Studio, Room DB, Material Design"
)

pdf.project(
    "Traffic Stability Analysis (LAR)",
    "",
    "Statistical analysis of Look-Ahead Range impact on traffic flow stability. "
    "Hypothesis testing, regression, and visualisation in R.",
    "R, ggplot2, plotly, rstatix"
)

pdf.project(
    "ETL & Data Warehousing with Power BI",
    "",
    "End-to-end ETL pipeline with Microsoft SSIS, OLAP cube, and Power BI dashboards for business analytics.",
    "SSIS, SQL Server, Power BI, OLAP Cube, Excel"
)

# Learning Projects
pdf.category_title("Learning Projects", "Practical projects built to learn and experiment")

pdf.project(
    "Titanic Survival Prediction",
    "",
    "End-to-end ML pipeline with experiment tracking, hyperparameter tuning, and reproducible pipelines.",
    "Random Forest, MLflow, DVC, Optuna, Hydra, DAGsHub"
)

pdf.project(
    "Algerian Forest Fire Prediction",
    "",
    "Regression models for predicting forest fire severity using linear, ridge, and elastic net regularisation.",
    "Linear Regression, Ridge, Elastic Net, Scikit-learn"
)

pdf.project(
    "PCA on Breast Cancer Dataset",
    "",
    "Principal Component Analysis for dimensionality reduction and visualisation on the Wisconsin breast cancer dataset.",
    "PCA, Scikit-learn, Matplotlib"
)

pdf.project(
    "Logistic Regression Practical",
    "",
    "Binary classification with logistic regression on a custom dataset, covering decision boundaries and evaluation metrics.",
    "Logistic Regression, Scikit-learn, NumPy"
)

pdf.project(
    "Used Car Price Prediction",
    "",
    "Regression model comparison: Random Forest, Linear, Ridge, Lasso, and Decision Tree. Random Forest performed best.",
    "Random Forest, Ridge, Lasso, Decision Tree, Scikit-learn"
)

pdf.project(
    "XGBoost Practices",
    "",
    "Hands-on practice with XGBoost for both classification and regression tasks.",
    "XGBoost, Scikit-learn, Pandas"
)

pdf.project(
    "Diabetes Prediction",
    "",
    "Predicting diabetes progression using Decision Tree Regressor on the sklearn diabetes dataset.",
    "Decision Tree, Scikit-learn, Pandas"
)

# Fun & Side Projects
pdf.category_title("Fun & Side Projects", "Personal experiments and passion projects")

pdf.project(
    "Adeesha Perera - Portfolio",
    "",
    "Minimalist developer portfolio with AI assistant, dark/light theme, onboarding, and interactive design elements.",
    "Next.js 16, TypeScript, Tailwind CSS v4, Framer Motion, RAG"
)

pdf.project(
    "Movie Recommendation System",
    "",
    "Comprehensive recommendation engine with KNN, SVD, SBERT content-based, and hybrid ensemble models on MovieLens data.",
    "Scikit-learn, Surprise, SBERT, Pandas, Jupyter"
)

pdf.project(
    "F1 Race Predictor",
    "",
    "Formula 1 race winner prediction using XGBoost regression on historical race data.",
    "XGBoost, Pandas, Scikit-learn"
)

output_path = "all_projects.pdf"
pdf.output(output_path)
print(f"Created {output_path}")
