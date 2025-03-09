# AI-Powered Alarm Diagnostic Assistant: Integrating BLIV and Vertex AI for Smart Industrial Maintenance with RAG
This project integrates Vertex AI Retrieval-Augmented Generation (RAG) with the BLIV monitoring dashboard to enhance industrial machine maintenance efficiency. The BLIV dashboard enables real-time monitoring of production flow and maintenance events, allowing manufacturers to track machine status continuously. In the event of a maintenance issue, The AI-powered RAG system rapidly diagnoses the root cause of the error and assists manufacturers in implementing solutions to restore the machine to optimal operation. Research published in IEEE Transactions on Industrial Informatics (Wang et al., 2023) demonstrates that integrating similar systems can reduce Mean Time to Repair (MTTR) by up to 60%, improve failure prediction accuracy by 75%, and optimize maintenance resource utilization by 40%.

# Project Description
## 1. **Real-time Error Detection with BLIV Dashboard**  
   ![Dashboard overall flow](https://github.com/nugrahazikry/spark-itb-rag-industry-maintenance/blob/main/maintenance-guidebook/dashboard%20overall%20flow.png)
   
   This project implements a real-time production flow monitoring dashboard integrated within the BLIV system, designed to detect errors at specific points in industrial machinery. By continuously monitoring production processes, the BLIV dashboard enables manufacturers to rapidly identify malfunctions or inefficiencies, reducing the risk of prolonged downtime. The system provides instant notifications and diagnostic insights, allowing maintenance teams to take immediate corrective action. This proactive approach enhances operational efficiency and ensures smoother production workflows.

## 2. **AI-Powered RAG as maintenance troubleshoot assistance** 
   ![RAG flow](https://github.com/nugrahazikry/spark-itb-rag-industry-maintenance/blob/main/maintenance-guidebook/rag%20document%20store.png)
   
   Once an error is detected through the BLIV dashboard, the issue is automatically relayed to the Retrieval-Augmented Generation (RAG) system, which is connected to an extensive database of maintenance documentation. The AI-powered RAG system analyzes the detected error, cross-references historical maintenance records, and provides precise troubleshooting guidance to assist manufacturers in diagnosing and resolving the issue efficiently. This integration significantly reduces the time required for problem resolution, minimizes human error, and enhances decision-making by leveraging AI-driven insights.

# Prototype methodology flow
## 1. Manufacturing production data generation pipeline
   ![Manufacturing data generation](https://github.com/nugrahazikry/spark-itb-rag-industry-maintenance/blob/main/maintenance-guidebook/generating%20pipeline.jpeg)
   
   To simulate the production flow of industrial machines, we have developed a data generation pipeline that replicates the operational behavior of machines within a factory setting. This synthetic manufacturing data serves as the foundation for testing and refining our error detection and maintenance optimization system. While the dataset is inspired by [OEE feature dashboard](https://evocon.com/feature/oee-dashboard/), we have made significant modifications to tailor it to our specific needs. These modifications enhance its applicability in automated error detection, ensuring that the system can recognize, classify, and respond to various machine failures and inefficiencies in real-time.

## 2. Error detection monitoring from BLIV dashboard
   ![BLIV dashboard monitoring](https://github.com/nugrahazikry/spark-itb-rag-industry-maintenance/blob/main/maintenance-guidebook/bliv%20dashboard.png)
   
   The BLIV dashboard provides an intuitive and comprehensive visualization of manufacturing machine performance, offering manufacturers real-time insights into their production processes. It features several key data representations, including:
   - **Overall Equipment Effectiveness (OEE)** – Displayed as a percentage, OEE quantifies machine performance by measuring availability, performance efficiency, and production quality. This metric helps manufacturers assess the effectiveness of their machinery and identify areas for improvement.
   - **Hourly Machine Status Trends** – A historical graph that tracks machine activity over time, allowing manufacturers to monitor operational patterns and potential anomalies.
   - **Alarm Condition Log** – A system that records error events and alert conditions, providing detailed information about the nature, frequency, and severity of machine issues.
   - **Machine Sensor & Alarm Log** – A detailed log capturing sensor readings and triggered alarms at specific time intervals, offering granular insights into machine health.
By leveraging this dashboard, manufacturers gain access to real-time operational intelligence, enabling them to detect irregularities promptly and make data-driven decisions for proactive maintenance interventions.

## 3. AI-powered LLM and RAG chatbot as troubleshoot assistant
   ![RAG UI](https://github.com/nugrahazikry/spark-itb-rag-industry-maintenance/blob/main/maintenance-guidebook/RAG%20example.png)
   
   To enhance maintenance efficiency and troubleshooting accuracy, we have integrated Vertex AI’s large language model (LLM) with Retrieval-Augmented Generation (RAG), deployed on Google Cloud Platform (GCP). This AI-powered system acts as an intelligent assistant, enabling manufacturers to retrieve contextually relevant information for diagnosing and resolving machine errors. The following is the detail on how it works:
   - **Corpus-Based Knowledge Retrieval** – The maintenance documents, technical manuals, and historical machine logs are pre-processed and stored in a vectorized corpus database within GCP. These documents serve as the primary knowledge source for the RAG model.
   - **RAG-Enhanced Response Generation** – When manufacturers request troubleshooting assistance, the system first retrieves the most relevant maintenance documents from the vector database before generating a response. This process ensures that the AI provides factually grounded and precise solutions, significantly reducing the likelihood of hallucinated (fabricated) responses.
   - **Accuracy Validation & Source Verification** – To further mitigate AI hallucination, the system provides: **RAG Result Accuracy Metrics with RAGAS** as a confidence score indicating how well the retrieved knowledge aligns with the query, and **The Source Source Document Image** as a direct reference to the original maintenance document, stored in Google Cloud Storage, allowing manufacturers to verify the AI-generated recommendations.

# List of Services used in this Project
![Architecture](https://github.com/nugrahazikry/spark-itb-rag-industry-maintenance/blob/main/maintenance-guidebook/architecture.png)

- **BLIV Data Pipeline**: A data generation framework designed to simulate real-world manufacturing production data, enabling accurate testing and validation of error detection and maintenance strategies in an industrial setting.
- **BLIV Dashboard**: A real-time visualization platform that transforms raw machine data from the BLIV pipeline into interactive charts and analytical reports, allowing manufacturers to monitor production efficiency, detect errors, and optimize performance.
- **GCP VM Instance**: A scalable computing infrastructure used to deploy and run real-time monitoring systems, ensuring seamless execution of AI-driven maintenance solutions.
- **Vertex AI Gemini**: LLM that powers the AI-driven troubleshooting assistant, facilitating natural language interactions, intelligent diagnostics, and automated decision support for industrial maintenance.
- **Vertex AI Corpus grounding**:  A knowledge retrieval framework that leverages a vectorized corpus database to enhance the accuracy of AI-generated responses, ensuring that troubleshooting recommendations are factually grounded in maintenance documents

# RAG Application Demo
You can try the application yourself here:
[RAG application demo](https://spark-itb-rag-industry-maintenance.streamlit.app/)

# Contributors
Contributors names and contact info: 
1. **[Muhammad Fikri Fadillah](https://github.com/boxside)**: Developed the BLIV data pipeline, creating a robust system that generates and simulates real-world manufacturing production data.
2. **[Diki Rustian](https://github.com/dikirust)**: Designed and implemented the BLIV Dashboard, focusing on intuitive chart representations to enhance monitoring and decision-making in manufacturing processes.
3. **[Zikry Adjie Nugraha](https://github.com/nugrahazikry)**: integrated the Vertex AI RAG solution, ensuring efficient knowledge retrieval and AI-driven troubleshooting for industrial maintenance applications.
