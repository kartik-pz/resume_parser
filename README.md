# Optimized CV Parsing API with FastAPI and OpenVINO

## 1. Application Overview

This project implements a scalable, high-performance REST API for parsing Curriculum Vitae (CVs)/resumes and extracting structured information such as names, contact information, skills, experience, education, etc.

The API is built using:
- **FastAPI:** For creating efficient, asynchronous API endpoints.
- **OpenVINO™ Toolkit:** For optimized, low-latency inference of the NLP model on CPU.
- **Hugging Face `transformers` & `optimum-intel`:** For leveraging a pre-trained resume NER model (`manishiitg/resume-ner`) and quantizing it to INT8 for further performance gains.
- **Docker:** For containerizing the application for consistent deployment and scalability.

The service accepts resume file uploads (PDF, DOCX, TXT), extracts text, processes it through the optimized NLP model, and returns the structured data in JSON format.

The pipeline is designed for efficiency:
- **Optimized Model:** An INT8 quantized OpenVINO version of the `manishiitg/resume-ner` model is used, specifically optimized for CPU inference, leading to lower latency.
- **Model Loading:** The model and tokenizer are loaded once at application startup to avoid repeated loading overhead for each request.
- **Asynchronous API:** FastAPI's asynchronous nature allows handling I/O-bound operations (like file uploads and network responses) efficiently without blocking.
- **Thread Pool for Blocking Tasks:** CPU-bound tasks such as text extraction from files (PDF/DOCX) and the NLP model inference itself are executed in a separate thread pool (`run_in_threadpool`). This prevents the main asynchronous event loop from being blocked, enabling the server to handle concurrent requests effectively.
- **Scalability:** To handle 1000+ concurrent requests, this FastAPI app would be deployed with multiple workers (e.g., Gunicorn + Uvicorn workers) behind a load balancer.
- **Efficient Text Extraction:** Libraries like `pypdfium2` (for PDF) and `python-docx` (for DOCX) are used for text extraction.

---