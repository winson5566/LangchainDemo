## 🛠️ Setup Instructions

> **Environment Requirement:**  
> Python **3.12** .

---

#  1.  Preparation
```bash
pip install -r requirements.txt
```

# 2. Indexing Documents
Add new documents directly into the data/doc/ directory, then run the following command to incrementally update the index:
```bash
python -m scripts.ingest
```

# 3. Start Backend
```pyhton
python -m backend.app
```

# 4. Start Frontend
```pyhton
streamlit run frontend/streamlit_app.py
```

# 5. Access the App    
Open your browser and visit: http://localhost:8501
