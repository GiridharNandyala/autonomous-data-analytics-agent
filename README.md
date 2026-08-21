# 📊 Autonomous Data Analytics Agent System

An end-to-end multi-agent Data Analytics pipeline built with **Streamlit**, **Python**, and **LLMs / Rule-Based Engines**. This application automates raw CSV data ingestion, performs intelligent data cleaning, conducts Exploratory Data Analysis (EDA), extracts strategic business insights, and automatically generates a downloadable PDF executive report.

## 🚀 Live Demo & Links

- 🌐 **Live Web Application:** [Autonomous Data Analytics Agent App](https://giridhar-autonomous-data-analytics-agent.streamlit.app/)
- 💼 **LinkedIn Announcement Post:** [View Project Demo on LinkedIn](https://www.linkedin.com/posts/giridhar-nandyala-5758662b2_python-dataanalytics-streamlit-ugcPost-7496476096247316480-jvER/)

---

## ✨ Key Features
- 📁 **Automated Data Ingestion**: Seamless handling of large CSV files (tested with 9,800+ rows and complex datasets).
- 🧹 **Smart Data Cleaning**: Automatic detection and handling of missing values, duplicates, and incorrect data types.
- 📈 **Exploratory Data Analysis (EDA)**: Interactive data visualizations, statistical distribution checks, and trend analysis.
- 🤖 **AI Analytics Crew**: Intelligent agents that extract actionable business insights and narrative recommendations.
- 📄 **Executive PDF Generation**: Generates a professional, downloadable PDF summary report at the click of a button.

---

## 🛠️ Tech Stack
- **Frontend & UI**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn, Plotly
- **AI & Logic**: OpenAI API / Rule-Based Insights Engine
- **Report Generation**: ReportLab / FPDF

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 to 3.12 installed
- Virtual environment configured

### Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
   
   1.Activate virtual environment & install dependencies:
   pip install -r requirements.txt

   2.Setup .env file for OpenAI API Key (Optional):
   OPENAI_API_KEY=your_openai_api_key_here

   3.Launch the application:
   streamlit run app.py
