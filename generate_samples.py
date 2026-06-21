"""Generate sample PDF documents for testing the RAG system."""
import os
from fpdf import FPDF


class SamplePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, self.title, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(5)

    def section(self, title, content):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, content)
        self.ln(3)


def create_resume_pdf():
    pdf = SamplePDF()
    pdf.title = "Alex Johnson - ML Engineer Resume"
    pdf.add_page()

    pdf.section("CONTACT", "Email: alex.johnson@email.com | Phone: (555) 123-4567\nLinkedIn: linkedin.com/in/alexjohnson | GitHub: github.com/alexjml")

    pdf.section("SKILLS",
        "Programming: Python, SQL, JavaScript, C++\n"
        "ML/DL: PyTorch, TensorFlow, scikit-learn, Hugging Face Transformers\n"
        "MLOps: Docker, Kubernetes, MLflow, Weights & Biases, Airflow\n"
        "Cloud: AWS (SageMaker, EC2, S3, Lambda), GCP (Vertex AI), Azure ML\n"
        "Databases: PostgreSQL, MongoDB, Redis, Pinecone\n"
        "Specialties: NLP, Computer Vision, Recommendation Systems, Time Series Forecasting")

    pdf.section("EXPERIENCE",
        "Senior ML Engineer | TechCorp Inc. | Jan 2022 - Present\n"
        "- Designed and deployed a real-time fraud detection system processing 50K transactions/sec\n"
        "- Reduced false positive rate by 35% using ensemble methods and feature engineering\n"
        "- Built automated ML pipeline with MLflow reducing model deployment time from days to hours\n"
        "- Led team of 4 engineers to migrate legacy models to cloud-native architecture\n\n"
        "ML Engineer | DataFlow Systems | Jun 2020 - Dec 2021\n"
        "- Developed NLP models for customer sentiment analysis achieving 94% F1 score\n"
        "- Implemented A/B testing framework for ML models serving 2M+ daily users\n"
        "- Created automated data validation pipeline reducing data quality issues by 60%\n\n"
        "Data Scientist | Analytics Pro | Aug 2018 - May 2020\n"
        "- Built predictive models for customer churn achieving 89% accuracy\n"
        "- Developed time series forecasting models for demand planning\n"
        "- Created interactive dashboards for business stakeholders using Tableau")

    pdf.section("EDUCATION",
        "Master of Science in Computer Science (Machine Learning)\nStanford University | 2018\n"
        "GPA: 3.9/4.0 | Thesis: Deep Learning for Sequential Recommendation\n\n"
        "Bachelor of Science in Computer Science\nUniversity of California, Berkeley | 2016\n"
        "GPA: 3.8/4.0 | Dean's List, ACM Programming Contest Finalist")

    pdf.section("PROJECTS",
        "1. FraudGuard - Real-time Fraud Detection\n"
        "   - Architecture: Event-driven microservices with Kafka, Redis feature store\n"
        "   - Tech: PyTorch, XGBoost, FastAPI, Docker, AWS SageMaker\n"
        "   - Impact: Detected $2.1M in fraudulent transactions in first quarter\n\n"
        "2. MedNLP - Clinical Text Understanding\n"
        "   - Fine-tuned BERT models for medical entity recognition and relation extraction\n"
        "   - Tech: Hugging Face Transformers, spaCy, PostgreSQL\n"
        "   - Impact: Automated extraction of 15K+ medical records with 96% accuracy\n\n"
        "3. RecSysPro - Recommendation Engine\n"
        "   - Hybrid collaborative filtering + content-based recommendation system\n"
        "   - Tech: PyTorch, FAISS, Redis, FastAPI\n"
        "   - Impact: Increased user engagement by 28% in A/B tests")

    pdf.section("PUBLICATIONS",
        "1. Johnson et al. 'Efficient Transformer Models for Real-time Fraud Detection' - NeurIPS 2023\n"
        "2. Johnson and Lee 'Hybrid Recommendation Systems with Graph Neural Networks' - KDD 2022")

    os.makedirs("data/resume", exist_ok=True)
    pdf.output("data/resume/resume_sample.pdf")
    print("Created: data/resume/resume_sample.pdf")


def create_project_pdf():
    pdf = SamplePDF()
    pdf.title = "FraudGuard - ML Project Documentation"
    pdf.add_page()

    pdf.section("Project Overview",
        "FraudGuard is a real-time fraud detection system designed for high-throughput "
        "financial transaction processing. The system analyzes transaction patterns, "
        "user behavior, and contextual features to identify potentially fraudulent "
        "activities in real-time with sub-100ms latency requirements.")

    pdf.section("Business Problem",
        "Financial fraud costs businesses over $50 billion annually. Traditional rule-based "
        "systems generate high false positive rates and miss sophisticated fraud patterns. "
        "FraudGuard uses machine learning to reduce false positives while catching more "
        "fraudulent transactions through adaptive pattern recognition.")

    pdf.section("Architecture",
        "The system follows a lambda architecture with both batch and streaming components:\n\n"
        "Data Ingestion Layer:\n"
        "- Apache Kafka for real-time transaction streaming\n"
        "- AWS S3 for historical data lake storage\n"
        "- Apache Airflow for batch ETL pipelines\n\n"
        "Feature Engineering:\n"
        "- Redis for real-time feature store (transaction velocity, amount patterns)\n"
        "- Feature computation pipeline with Apache Flink\n"
        "- Feature versioning with MLflow\n\n"
        "Model Serving:\n"
        "- FastAPI endpoints behind load balancer\n"
        "- Model ensemble: XGBoost + Neural Network voting\n"
        "- A/B testing framework for model comparison\n\n"
        "Monitoring:\n"
        "- Prometheus + Grafana for system metrics\n"
        "- Custom dashboards for model performance\n"
        "- Automated alerting for drift detection")

    pdf.section("Technical Implementation",
        "Feature Engineering:\n"
        "- Transaction amount z-scores per user (rolling 30-day window)\n"
        "- Transaction frequency features (hourly, daily, weekly patterns)\n"
        "- Merchant category embeddings\n"
        "- Geographic distance from user's typical locations\n"
        "- Device fingerprint and IP geolocation features\n"
        "- Time-since-last-transaction features\n\n"
        "Model Architecture:\n"
        "- Primary: XGBoost with custom objective function (weighted cross-entropy)\n"
        "- Secondary: 1D-CNN for sequential transaction patterns\n"
        "- Ensemble: Soft voting with learned weights\n"
        "- Threshold optimization using cost-sensitive metrics\n\n"
        "Training Pipeline:\n"
        "- Weekly retraining on latest 90 days of data\n"
        "- SMOTE for handling class imbalance (0.1% fraud rate)\n"
        "- Hyperparameter tuning with Optuna\n"
        "- Model validation with stratified time-series split")

    pdf.section("Results",
        "Performance Metrics:\n"
        "- Precision: 0.92 (up from 0.78 with rule-based system)\n"
        "- Recall: 0.95 (up from 0.85)\n"
        "- F1 Score: 0.935\n"
        "- AUC-ROC: 0.987\n"
        "- False Positive Rate: Reduced by 62%\n\n"
        "Business Impact:\n"
        "- Detected $2.1M in fraudulent transactions in first quarter\n"
        "- Reduced manual review workload by 45%\n"
        "- Average detection latency: 47ms\n"
        "- System uptime: 99.97%")

    pdf.section("Challenges and Solutions",
        "1. Class Imbalance: Only 0.1% of transactions are fraudulent\n"
        "   Solution: Combined SMOTE oversampling with cost-sensitive learning and custom loss functions\n\n"
        "2. Real-time Requirements: Sub-100ms response time needed\n"
        "   Solution: Redis feature store, model optimization with ONNX, batch prediction for non-critical paths\n\n"
        "3. Concept Drift: Fraud patterns evolve constantly\n"
        "   Solution: Automated drift detection, weekly retraining, champion-challenger model deployment\n\n"
        "4. Interpretability: Regulatory requirements for explainable decisions\n"
        "   Solution: SHAP values for feature importance, natural language explanations for alerts")

    pdf.section("Lessons Learned",
        "- Start with simple baselines before complex models\n"
        "- Feature engineering often matters more than model architecture\n"
        "- Production ML requires significant infrastructure investment\n"
        "- Monitoring and observability are as important as model performance\n"
        "- Cross-functional collaboration with business stakeholders is critical")

    os.makedirs("data/completed_ml_projects", exist_ok=True)
    pdf.output("data/completed_ml_projects/fraudguard_project.pdf")
    print("Created: data/completed_ml_projects/fraudguard_project.pdf")


def create_mednlp_pdf():
    pdf = SamplePDF()
    pdf.title = "MedNLP - Clinical Text Understanding"
    pdf.add_page()

    pdf.section("Project Overview",
        "MedNLP is a clinical natural language processing system designed to extract "
        "structured information from unstructured medical records. The system performs "
        "named entity recognition (NER), relation extraction, and clinical coding to "
        "automate the processing of over 15,000 medical documents daily.")

    pdf.section("Problem Statement",
        "Healthcare organizations generate millions of unstructured clinical notes annually. "
        "Manual processing is time-consuming, expensive, and prone to errors. MedNLP "
        "automates extraction of diagnoses, medications, procedures, and their relationships "
        "from clinical text, enabling downstream analytics and decision support systems.")

    pdf.section("Technical Approach",
        "Model Architecture:\n"
        "- Base: ClinicalBERT (BioBERT fine-tuned on MIMIC-III clinical notes)\n"
        "- NER: Token classification head with CRF layer\n"
        "- Relation Extraction: Entity-aware attention mechanism\n"
        "- Clinical Coding: Multi-label classification for ICD-10 codes\n\n"
        "Training Data:\n"
        "- MIMIC-III/IV clinical notes (de-identified)\n"
        "- i2b2/VA challenge datasets for NER\n"
        "- Custom annotated dataset of 5,000 discharge summaries\n\n"
        "Entity Types Extracted:\n"
        "- Problem: Diseases, symptoms, diagnoses\n"
        "- Treatment: Medications, procedures, therapies\n"
        "- Test: Lab results, imaging studies\n"
        "- Person: Patients, providers\n"
        "- Temporal: Dates, durations, frequencies\n\n"
        "Relation Types:\n"
        "- Treats: Medication treats condition\n"
        "- Causes: Condition causes symptom\n"
        "- Indicates: Test indicates diagnosis\n"
        "- Administers: Provider administers treatment")

    pdf.section("Implementation",
        "Data Pipeline:\n"
        "- De-identification module for PHI removal\n"
        "- Text preprocessing (section detection, abbreviation expansion)\n"
        "- Annotation pipeline with Prodigy for active learning\n\n"
        "Model Training:\n"
        "- Mixed precision training with PyTorch Lightning\n"
        "- Gradient accumulation for large batch sizes\n"
        "- Early stopping based on validation F1\n\n"
        "Deployment:\n"
        "- ONNX Runtime for optimized inference\n"
        "- Docker containers on Kubernetes\n"
        "- Batch processing for non-real-time workloads\n"
        "- REST API for real-time queries")

    pdf.section("Results",
        "NER Performance:\n"
        "- Problem entities: F1 = 0.94\n"
        "- Treatment entities: F1 = 0.92\n"
        "- Test entities: F1 = 0.91\n"
        "- Overall NER F1: 0.93\n\n"
        "Relation Extraction:\n"
        "- Treats relations: F1 = 0.89\n"
        "- Causes relations: F1 = 0.86\n"
        "- Overall RE F1: 0.88\n\n"
        "Clinical Coding:\n"
        "- Micro F1: 0.81\n"
        "- Macro F1: 0.74\n"
        "- Top-10 accuracy: 0.92")

    pdf.section("Impact",
        "- Automated processing of 15,000+ documents daily\n"
        "- Reduced manual chart review time by 70%\n"
        "- Enabled real-time clinical decision support alerts\n"
        "- Improved quality metrics reporting accuracy by 35%")

    os.makedirs("data/completed_ml_projects", exist_ok=True)
    pdf.output("data/completed_ml_projects/mednlp_project.pdf")
    print("Created: data/completed_ml_projects/mednlp_project.pdf")


def create_fun_projects_pdf():
    pdf = SamplePDF()
    pdf.title = "Adeesha - Fun Projects"
    pdf.add_page()

    pdf.section("Project 1: AI Music Generator",
        "An AI-powered music composition tool that generates original melodies "
        "based on user input moods and styles.\n\n"
        "Tech Stack: Python, TensorFlow, MIDI processing, Flask\n"
        "Features:\n"
        "- Mood-based music generation (happy, sad, energetic)\n"
        "- Multiple genre support (jazz, classical, lo-fi)\n"
        "- Real-time playback and MIDI export\n"
        "- Web interface with interactive controls\n\n"
        "Duration: 2 weeks\n"
        "Status: Completed\n"
        "GitHub: github.com/adeesha/ai-music-gen")

    pdf.section("Project 2: Retro Game AI Bot",
        "A reinforcement learning agent that learns to play classic arcade games "
        "using Deep Q-Networks.\n\n"
        "Tech Stack: Python, PyTorch, OpenAI Gym, Atari\n"
        "Features:\n"
        "- Self-learning agent using DQN algorithm\n"
        "- Experience replay and target network\n"
        "- Training visualization with TensorBoard\n"
        "- Score tracking and model checkpoints\n\n"
        "Results:\n"
        "- Achieved human-level performance on Breakout\n"
        "- Trained 10M frames in 4 hours on single GPU\n\n"
        "Duration: 3 weeks\n"
        "Status: Completed")

    pdf.section("Project 3: Smart Recipe Generator",
        "An NLP-powered app that generates recipes based on available ingredients.\n\n"
        "Tech Stack: Python, Hugging Face Transformers, FastAPI, React\n"
        "Features:\n"
        "- Ingredient-based recipe suggestions\n"
        "- Dietary restriction filters\n"
        "- Step-by-step cooking instructions\n"
        "- Nutritional information calculator\n\n"
        "Duration: 1 week\n"
        "Status: Completed")

    os.makedirs("data/inprogress_ml_projects", exist_ok=True)
    pdf.output("data/inprogress_ml_projects/fun_projects.pdf")
    print("Created: data/inprogress_ml_projects/fun_projects.pdf")


def create_campus_projects_pdf():
    pdf = SamplePDF()
    pdf.title = "Adeesha - Campus Projects"
    pdf.add_page()

    pdf.section("Project 1: Campus Event Management System",
        "A full-stack web application for managing university events, "
        "registrations, and attendance tracking.\n\n"
        "Tech Stack: Django, PostgreSQL, React, Docker\n"
        "Features:\n"
        "- Event creation and management\n"
        "- Student registration and QR code check-in\n"
        "- Real-time attendance tracking\n"
        "- Analytics dashboard for organizers\n"
        "- Email notifications and reminders\n\n"
        "Team Size: 4 members\n"
        "Role: Backend Lead\n"
        "Duration: 4 months\n"
        "Grade: A+")

    pdf.section("Project 2: Distributed File Storage System",
        "A Google Drive-like file storage system with chunking, replication, "
        "and fault tolerance.\n\n"
        "Tech Stack: Go, gRPC, Redis, MinIO\n"
        "Features:\n"
        "- File upload/download with chunking\n"
        "- Data replication across nodes\n"
        "- Automatic failover and recovery\n"
        "- Access control and sharing permissions\n"
        "- Web interface for file management\n\n"
        "Team Size: 3 members\n"
        "Role: System Architect\n"
        "Duration: 3 months\n"
        "Grade: A")

    pdf.section("Project 3: Real-time Chat Application",
        "A WhatsApp-like messaging app with end-to-end encryption.\n\n"
        "Tech Stack: Node.js, Socket.io, MongoDB, React Native\n"
        "Features:\n"
        "- Real-time messaging with WebSocket\n"
        "- End-to-end encryption (Signal Protocol)\n"
        "- Group chats and media sharing\n"
        "- Online status and read receipts\n"
        "- Message search and archive\n\n"
        "Team Size: 2 members\n"
        "Role: Full Stack Developer\n"
        "Duration: 2 months\n"
        "Grade: A+")

    os.makedirs("data/uni_projects", exist_ok=True)
    pdf.output("data/uni_projects/campus_projects.pdf")
    print("Created: data/uni_projects/campus_projects.pdf")


def create_portfolio_projects_pdf():
    pdf = SamplePDF()
    pdf.title = "Adeesha - Portfolio Projects"
    pdf.add_page()

    pdf.section("Project 1: ML-Powered Resume Scanner",
        "An AI tool that parses resumes and matches candidates to job descriptions "
        "using NLP and semantic similarity.\n\n"
        "Tech Stack: Python, spaCy, Sentence Transformers, FastAPI, PostgreSQL\n"
        "Features:\n"
        "- Resume parsing (PDF, DOCX)\n"
        "- Skill extraction and matching\n"
        "- Semantic similarity scoring\n"
        "- Batch processing for recruiters\n"
        "- REST API for integration\n\n"
        "Impact:\n"
        "- Reduced screening time by 80%\n"
        "- Processed 10,000+ resumes\n"
        "- 95% accuracy on skill matching\n\n"
        "Live Demo: resume-scanner.adeesha.dev\n"
        "GitHub: github.com/adeesha/resume-scanner")

    pdf.section("Project 2: Sentiment Analysis Dashboard",
        "A real-time dashboard analyzing social media sentiment about brands "
        "and products.\n\n"
        "Tech Stack: Python, Transformers, Kafka, React, D3.js\n"
        "Features:\n"
        "- Real-time tweet analysis\n"
        "- Sentiment trend visualization\n"
        "- Alert system for negative sentiment\n"
        "- Competitor comparison analytics\n"
        "- Export reports (PDF, CSV)\n\n"
        "Impact:\n"
        "- Monitored 100K+ tweets daily\n"
        "- Detected PR crises 6 hours faster\n"
        "- Used by 3 marketing teams\n\n"
        "Live Demo: sentiment.adeesha.dev")

    pdf.section("Project 3: Personal Finance Tracker",
        "An intelligent expense tracker with automated categorization "
        "and budget recommendations.\n\n"
        "Tech Stack: Python, scikit-learn, React Native, Firebase\n"
        "Features:\n"
        "- Bank statement parsing (CSV, PDF)\n"
        "- Auto-categorization using ML\n"
        "- Budget forecasting and alerts\n"
        "- Investment portfolio tracking\n"
        "- Monthly financial reports\n\n"
        "Impact:\n"
        "- 500+ active users\n"
        "- Saved users average 15% on expenses\n"
        "- 4.8/5 app store rating\n\n"
        "App Store: Available on iOS and Android")

    os.makedirs("data/inprogress_ml_projects", exist_ok=True)
    pdf.output("data/inprogress_ml_projects/portfolio_projects.pdf")
    print("Created: data/inprogress_ml_projects/portfolio_projects.pdf")


def create_practical_projects_pdf():
    pdf = SamplePDF()
    pdf.title = "Adeesha - Practical Projects"
    pdf.add_page()

    pdf.section("Project 1: Customer Churn Prediction System",
        "A production ML system predicting customer churn for a SaaS company "
        "with automated retraining pipeline.\n\n"
        "Tech Stack: Python, XGBoost, Airflow, MLflow, AWS\n"
        "Business Problem:\n"
        "- 15% monthly churn rate costing $500K annually\n"
        "- No visibility into at-risk customers\n\n"
        "Solution:\n"
        "- Feature engineering from user behavior logs\n"
        "- XGBoost model with 89% recall\n"
        "- Automated daily predictions\n"
        "- Email alerts for high-risk customers\n\n"
        "Impact:\n"
        "- Reduced churn by 22% in 3 months\n"
        "- Saved $110K in first quarter\n"
        "- ROI: 5x investment\n\n"
        "Client: SaaS Startup (NDA)")

    pdf.section("Project 2: Inventory Optimization System",
        "An ML-powered inventory management system for retail chain "
        "optimizing stock levels and reorder points.\n\n"
        "Tech Stack: Python, Prophet, PostgreSQL, Django, Docker\n"
        "Business Problem:\n"
        "- $2M annual loss from overstocking/understocking\n"
        "- Manual ordering based on gut feeling\n\n"
        "Solution:\n"
        "- Time series forecasting for demand prediction\n"
        "- Safety stock optimization\n"
        "- Automated reorder recommendations\n"
        "- Dashboard for inventory managers\n\n"
        "Impact:\n"
        "- Reduced inventory costs by 18%\n"
        "- Improved stock availability to 98%\n"
        "- Freed up $400K in working capital\n\n"
        "Client: Retail Chain (50 stores)")

    pdf.section("Project 3: Document Processing Pipeline",
        "An automated document extraction system for insurance claims "
        "processing.\n\n"
        "Tech Stack: Python, Tesseract, Transformers, FastAPI, Redis\n"
        "Business Problem:\n"
        "- Manual processing of 1000+ claims daily\n"
        "- Average processing time: 15 minutes per claim\n"
        "- High error rate in data entry\n\n"
        "Solution:\n"
        "- OCR for document digitization\n"
        "- NLP for field extraction\n"
        "- Validation rules engine\n"
        "- Integration with claims management system\n\n"
        "Impact:\n"
        "- Reduced processing time to 2 minutes per claim\n"
        "- Error rate reduced by 85%\n"
        "- Saved 3 FTE equivalent\n\n"
        "Client: Insurance Company")

    os.makedirs("data/inprogress_ml_projects", exist_ok=True)
    pdf.output("data/inprogress_ml_projects/practical_projects.pdf")
    print("Created: data/inprogress_ml_projects/practical_projects.pdf")


if __name__ == "__main__":
    print("Generating sample PDFs...")
    create_resume_pdf()
    create_project_pdf()
    create_mednlp_pdf()
    create_fun_projects_pdf()
    create_campus_projects_pdf()
    create_portfolio_projects_pdf()
    create_practical_projects_pdf()
    print("\nDone! 7 PDF files created in data/")
